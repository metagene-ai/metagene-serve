# https://github.com/huggingface/peft/blob/main/examples/sft/train.py
import csv
from dataclasses import dataclass, field
import json
import logging
import numpy as np
import os
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
import sys
import transformers
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          PreTrainedTokenizerFast,
                          BitsAndBytesConfig,
                          Trainer, TrainingArguments, HfArgumentParser, set_seed)
import torch
from torch.utils.data import Dataset
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union


def clear_gpu_cache(rank=None):
    if rank == 0:
        print("Clearing GPU cache for all ranks.")
    torch.cuda.empty_cache()

# https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py#L111
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

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

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True)

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

# https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py#L168
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(valid_labels, valid_predictions),
        "precision": precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

# Some notes about models choices
#  we have either z2 and z3 from deeepspeed for optimization
#   z2 have supports for bnb quantization
#   z3 should also have according to the following links but somehow does not have supports for llama 3.X multi modality models using zero.init()
#     https://github.com/huggingface/accelerate/issues/1228
#     https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed
#  Given the above constraints and we use multi modality model:
#  if we want to do half-precision finetuning, we can use z2 or z3 with llama 3.2-11B-Vision-Instruct, llama 3.2-90B-Vision-Instruct
#  if we want to do quantized finetuning, we can use z2 or z3 (no zero.init) with llama 3.2-11B-Vision-Instruct, llama 3.2-90B-Vision-Instruct
def get_model_init_fn(model_args, num_labels):

    def _model_init():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config if model_args.use_4bit_quantization else None,
            num_labels=num_labels,
            # attn_implementation="eager",  # "flash_attention_2" is not for V100 :(
            attn_implementation="flash_attention_2",
            torch_dtype = torch.bfloat16
        ).to(torch.device("cuda"))
        model.config._name_or_path = model_args.model_name_or_path

        peft_config = LoraConfig(
            r=model_args.r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.target_modules.split(","),
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
            # modules_to_save=["classifier", "score"]
        )
        model = get_peft_model(model, peft_config)

        # SFTTrainer in the current version of trl is not compatible with transformers 4.45.3
        # https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L210C1-L217C101
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model
    return _model_init

def get_split_tokenized_dataset(tokenizer, data_args):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=os.path.join(data_args.data_dir, "train.csv"))
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=os.path.join(data_args.data_dir, "dev.csv"))
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=os.path.join(data_args.data_dir, "test.csv"))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return data_collator, train_dataset, eval_dataset, test_dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model_ckpts/safetensors/step-00086000")
    use_4bit_quantization: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    r: Optional[int] = field(default=64)
    target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj")
    model_max_length: Optional[int] = field(default=512)

@dataclass
class DataArguments:
    data_dir: Optional[str] = field(default="./assets/data/GUE/EMP/H3")

@dataclass
class UtilsArguments:
    trial_output_file: Optional[str] = field(default="./trial_output_file.json")
    trial_number: Optional[int] = field(default=0)


def main():
    # get args and set seed
    model_args, data_args, training_args, utils_args = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, UtilsArguments)).parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # clear GPU cache
    clear_gpu_cache(training_args.local_rank)

    # tokenizer
    if training_args.local_rank == 0:
        print(f"\nLoading tokenizer from {model_args.model_name_or_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        return_tensors="pt")

    # tokenized dataset
    if training_args.local_rank == 0:
        print(f"\nTokenizing and splitting dataset {data_args.data_dir}")
    data_collector, train_dataset, eval_dataset, test_dataset = get_split_tokenized_dataset(tokenizer, data_args)

    # trainer
    if training_args.local_rank == 0:
        print(f"\nPreparing the HF trainer")
    if training_args.local_rank == 0 and model_args.use_4bit_quantization:
        print(f"\nModel will be quantized")
    trainer = Trainer(
        model_init=get_model_init_fn(model_args, train_dataset.num_labels),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collector)

    # train
    if training_args.local_rank == 0:
        print(f"\nStart training for trial {utils_args.trial_number}")
    trainer.train()

    # evaluate
    if training_args.local_rank == 0:
        print(f"\nEvaluating model for ...")
    eval_metrics = trainer.evaluate(test_dataset)

    with open(utils_args.trial_output_file, 'w') as f:
        trial_output = {
            'trial_number': utils_args.trial_number,
            'mcc': eval_metrics['eval_mcc']
            # 'f1': eval_metrics['eval_f1']
        }
        json.dump(trial_output, f, indent=4)

    # Reset for the next trial
    if training_args.local_rank == 0:
        print("\nClearing GPU cache at the end for the next trial")
    del trainer
    clear_gpu_cache(training_args.local_rank)


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()