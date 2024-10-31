import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
import torch
from peft import LoraConfig, get_peft_model

import wandb



# Load the datasets
train_df = pd.read_csv("/workspace/MGFM/data/fine-tune/GUE/EMP/H3/train.csv")
val_df = pd.read_csv("/workspace/MGFM/data/fine-tune/GUE/EMP/H3/dev.csv")
test_df = pd.read_csv("/workspace/MGFM/data/fine-tune/GUE/EMP/H3/test.csv")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the pre-trained model and tokenizer
model_path = "/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = 0

# Load the model in 4bit
nf4_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=nf4_quant_config,
    num_labels=len(train_df['label'].unique()),
    device_map="auto"
)

# Apply the LoRA adapter
# causal LM lora setting
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# sequence classification setting
# peft_config = LoraConfig(
#     task_type="SEQ_CLS",
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     bias="none",
#     inference_mode=False
# )
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

learning_rate=2e-5

# Define training arguments
training_args = TrainingArguments(
    run_name=f"test_run_{learning_rate}_{peft_config.task_type}",
    output_dir="./results",
    num_train_epochs=3,
    optim="adamw_torch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    warmup_steps=50,
    learning_rate=learning_rate,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    seed=42
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Init the wandb
wandb.init(project="MGFM-GUE-benchmark", name=training_args.run_name)

# Train the model
trainer.train()

# Evaluate on validation set
val_results = trainer.evaluate()

# Evaluate on test set
test_results = trainer.evaluate(tokenized_test)

# Print results
# print("Validation Results:", val_results)
print("Test Results:", test_results)

wandb.log(test_results)
wandb.finish()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Save metrics to a file
with open("evaluation_results.txt", "w") as f:
    f.write("Validation Results:\n")
    for key, value in val_results.items():
        f.write(f"{key}: {value}\n")
    f.write("\nTest Results:\n")
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")
