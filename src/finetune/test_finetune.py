import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
import torch

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
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, model_max_length=128)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = 0

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(train_df['label'].unique()))

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    mcc = matthews_corrcoef(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "mcc": mcc
    }

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on validation set
val_results = trainer.evaluate()

# Evaluate on test set
test_results = trainer.evaluate(tokenized_test)

# Print results
print("Validation Results:", val_results)
print("Test Results:", test_results)

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