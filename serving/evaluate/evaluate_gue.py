from datasets import Dataset
import numpy as np
import os
import optuna
import pandas as pd
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig


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
    r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = trial.suggest_categorical("target_modules", [
        ["q_proj", "k_proj"],
        ["q_proj", "v_proj"],
        ["k_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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

    # Update training arguments with the trial name
    args.run_name = trial_name
    args.output_dir = f"./results/{trial_name}"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
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

def main():
    import wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project="MGFM-GUE-benchmark", name=os.getenv('STUDY_ID'))

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
