import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import \
    PreTrainedTokenizerFast, \
    AutoTokenizer, \
    AutoModelForSequenceClassification, \
    TrainingArguments, \
    Trainer, \
    BitsAndBytesConfig

from sklearn.metrics import \
    accuracy_score, \
    recall_score, \
    f1_score, \
    matthews_corrcoef, \
    precision_score
from peft import \
    LoraConfig, \
    get_peft_model
import optuna

import os
import wandb
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)
wandb.init(project="MGFM-GUE-benchmark", name=os.getenv('STUDY_ID'))


# Adopted from the code base of Benchmarking DNA Foundation Models for Genomic Sequence Classification
# https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/job_scripts/inference_dnabert2.py
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Move inputs to the device
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs[0]  # Last hidden states

            # Apply mean pooling over token embeddings
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled_embeddings = torch.sum(attention_mask * hidden_states, axis=1) / torch.sum(attention_mask, axis=1)

        # Return as required by the Trainer (loss is None in this case)
        return (None, pooled_embeddings, labels)


# class MeanTokenEmbeddingModel(nn.Module):
#     def __init__(self, base_model, num_labels):
#         super(MeanTokenEmbeddingModel, self).__init__()
#         self.base_model = base_model
#         self.embedding_layer = base_model.get_input_embeddings()
#         self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None):
#         # Calculate mean token embedding
#         token_embeddings = self.embedding_layer(input_ids)  # Shape: (batch_size, sequence_length, hidden_size)
#         mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
#         sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
#         mean_token_embedding = sum_embeddings / sum_mask
#
#         # Pass mean embedding through transformer model
#         transformer_output = self.base_model(inputs_embeds=mean_token_embedding.unsqueeze(1),
#                                              attention_mask=torch.ones_like(attention_mask), output_hidden_states=True)
#
#         # Extract the last hidden state
#         last_hidden_state = transformer_output.hidden_states[-1][:, 0, :]  # First token of the last layer
#
#         # Classification on the pooled output
#         logits = self.classifier(last_hidden_state)  # Shape: (batch_size, num_labels)
#
#         # Compute loss if labels are provided
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
#
#         return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class MeanPoolingLlamaModel(nn.Module):
    def __init__(self, model):
        super(MeanPoolingLlamaModel, self).__init__()
        self.model = model
        # Define a classifier layer if not present in the model
        if not hasattr(model, "score"):  # Check if the classification layer is missing
            self.classifier = nn.Linear(model.config.hidden_size, model.config.num_labels)
        else:
            self.classifier = model.score  # Use the existing 'score' layer if it exists

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Pass inputs through the base model
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract the last hidden states
        last_hidden_state = outputs.last_hidden_state

        # Apply mean pooling, masking padding tokens
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Pass pooled output through the classifier layer
        logits = self.classifier(mean_pooled)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # Always return (loss, logits) for consistency
        return (loss, logits)


def tokenize_function(examples):
    return tokenizer(
        examples["sequence"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Exclude padding tokens (-100)
    non_padding_mask = (labels != -100)
    non_padding_predictions = predictions[non_padding_mask]
    non_padding_labels = labels[non_padding_mask]
    return {
        "accuracy": accuracy_score(non_padding_labels, non_padding_predictions),
        "f1": f1_score(non_padding_labels, non_padding_predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(non_padding_labels, non_padding_predictions),
        "precision": precision_score(non_padding_labels, non_padding_predictions, average="macro", zero_division=0),
        "recall": recall_score(non_padding_labels, non_padding_predictions, average="macro", zero_division=0),
    }


def objective(base_model, args, trial):
    # Define the search space
    # r = trial.suggest_categorical("r", [8, 16, 32, 64])
    # lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
    # lora_dropout = trial.suggest_categorical("lora_dropout", [0.05, 0.1])
    r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = trial.suggest_categorical("target_modules", [
        # ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj"],
        # ["q_proj", "k_proj", "v_proj", "o_proj"],
        # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ])

    trial_name = f"trial_{trial.number}_r{r}_alpha{lora_alpha}_dropout{lora_dropout}"
    wandb.run.name = trial_name
    wandb.run.save()

    # Create LoRA config
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
        inference_mode=False,
        # modules_to_save=["classifier", "score"]
    )

    # Create model, trainer, etc.
    model = get_peft_model(base_model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Wrap the model with MeanTokenEmbeddingModel
    model = MeanPoolingLlamaModel(model)
    # model = MeanTokenEmbeddingModel(model, num_labels=len(train_df['label'].unique()))

    # Update training arguments with the trial name
    args.run_name = trial_name
    args.output_dir = f"./results/{trial_name}"

    # Create Trainer instance
    # trainer = CustomTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=tokenized_train,
    #     eval_dataset=tokenized_val,
    #     compute_metrics=compute_metrics,
    # )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_test)

    # Log the trial results
    wandb.log({
        "trial": trial.number,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": str(target_modules),
        "test_mcc": test_results["eval_mcc"],
        "test_accuracy": test_results["eval_accuracy"],
        "test_f1": test_results["eval_f1"],
        "test_precision": test_results["eval_precision"],
        "test_recall": test_results["eval_recall"],
    })

    # Return negative MCC (since Optuna minimizes by default)
    return -test_results["eval_mcc"]


if __name__ == "__main__":
    # Load the datasets
    print("Loading datasets ...")
    train_df = pd.read_csv("../data/fine-tune/GUE/EMP/H3/train.csv")
    val_df = pd.read_csv("../data/fine-tune/GUE/EMP/H3/dev.csv")
    test_df = pd.read_csv("../data/fine-tune/GUE/EMP/H3/test.csv")
    print("Datasets loaded")

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load the model in 4bit
    print("Loading model ...")
    model_path = "../model_ckpts/safetensors/step-00078000"
    # nf4_quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        # quantization_config=nf4_quant_config,
        num_labels=len(train_df['label'].unique()),
        device_map="auto"
    )
    print("Model loaded")

    # Load the tokenizer
    print("Loading tokenizer & tokenizing the datasets ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0
    # Tokenize the datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.getenv('FINETUNE_OUTPUT_DIR'),
        logging_dir=os.getenv('FINETUNE_LOG_DIR'),
        num_train_epochs=2,
        # optim="adamw_torch",
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=6e-5,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16=True,
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        seed=42
    )

    # Create and run the study
    print("Starting the hp study ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(model, training_args, trial), n_trials=1)
    print("Study finished")

    best_trial = study.best_trial
    wandb.log({
        "best_trial_number": best_trial.number,
        "best_trial_mcc": -best_trial.value,
        "best_trial_params": best_trial.params
    })
    wandb.finish()

    print("Best trial:")
    print(f"  MCC: {-best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
