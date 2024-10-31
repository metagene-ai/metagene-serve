import os, csv, json, logging
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from dataclasses import dataclass, field

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizerFast
from transformers import TrainingArguments as HfTrainingArguments

import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score

# import wandb


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py
class SupervisedDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model")


@dataclass
class DataArguments:
    data_path: str = field(default=None)


@dataclass
class TrainingArguments(HfTrainingArguments):
    seed: int = field(default=42)
    run_name: str = field(default="run")
    output_dir: str = field(default="output")

    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=3e-5)
    weight_decay: float = field(default=0.01)
    num_train_epochs: float = field(default=3)
    model_max_length: int = field(default=128)
    warmup_steps: int = field(default=50)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    
    eval_and_save_results: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    save_steps: int = field(default=200)
    save_model: bool = field(default=True)
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=100000)
    log_level: Optional[str] = field(default="info")
    # report_to: str = field(default="wandb")
    report_to: str = field(default=None)
    
    fp16: bool = field(default=False)
    find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)
    checkpointing: bool = field(default=False)
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDatasetFast:
    tokenizer: transformers.PreTrainedTokenizerFast

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # Convert input_ids to a list of lists if they are not already
        if isinstance(input_ids[0], torch.Tensor):
            input_ids = [ids.tolist() for ids in input_ids]
        
        # Use the pad method of PreTrainedTokenizerFast
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt"
        )
        
        # Convert labels to tensor
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch


# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Exclude padding tokens (-100)
    non_padding_mask = (labels != -100)  
    non_padding_predictions = predictions[non_padding_mask]
    vnon_padding_labels = labels[non_padding_mask]
    return {
        "accuracy": accuracy_score(vnon_padding_labels, non_padding_predictions),
        "f1": f1_score(vnon_padding_labels, non_padding_predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(vnon_padding_labels, non_padding_predictions),
        "precision": precision_score(vnon_padding_labels, non_padding_predictions, average="macro", zero_division=0),
        "recall": recall_score(vnon_padding_labels, non_padding_predictions, average="macro", zero_division=0),
    }


def main():
    # parse the args
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # # Initialize wandb
    # if training_args.report_to == "wandb":
    #     wandb.init(project="MGFM-GUE-benchmark", name=training_args.run_name)

    # load the tokenizer
    print("Loading the tokenizer ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_args.model_name_or_path, 
        model_max_length=training_args.model_max_length)
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0
    print("Tokenizer loaded")

    # define the data collator and datasets
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetFast(tokenizer=tokenizer)
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "train.csv"))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "test.csv"))

    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the model ...")
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, use_safetensors=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = train_dataset.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        use_safetensors=True
    )
    model = model.to(device)
    print("Model loaded")
    
    # define the trainer
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator)
    
    # fine-tune the model and save results
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        state_dict = trainer.model.state_dict()

        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_path=training_args.output_path, state_dict=cpu_state_dict)


    # get the evaluation results
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_path, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)
        
        # if training_args.report_to == "wandb":
        #     # Log final results to wandb
        #     wandb.log(results)
        #     wandb.finish()


if __name__ == "__main__":
    main()
