# Load Dataset
import random
from collections import Counter
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
import argparse
import time

# from llama_cpp import Llama

# TO USE THIS SCRIPT, PLEASE CHANGE THE FOLLOWING DIRECTORIES
parser = argparse.ArgumentParser(description="Multi GPU test model format")
parser.add_argument("--model_format", type=str, required=True, help="Model format for multi gpu test")
args = parser.parse_args()
if args.model_format == "st":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000/"
elif args.model_format == "st_gptq":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/gptq_safetensors/step-00078000/"
elif args.model_format == "st_nf4":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/nf4_safetensors/step-00078000/"
# elif args.model_format == "gguf":
#     CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_gguf/step-00078000/"

DATASET_DIR = "/workspace/MGFM/data/sanity_check/cleaned_tokens_2000000000.txt"
TOKENIZER_CKPT_DIR = "/workspace/MGFM/model_ckpts/step-00078000/"
SINGLE_GPU = 1

N = 1000  # Dataset Size
B = 32  # Batch Size
CTX_LEN = 12  # Context Length
GEN_LEN = 20  # Generation Length

print("\n\n\nSTART: Loading Dataset")
random.seed(42)

# Multi-gpu inference set up
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    multi_gpu = True
else:
    print("Using single GPU")
    multi_gpu = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
with open(DATASET_DIR, "r") as f:
    i = 0
    for line in f:
        dataset.append("_" + line.strip())
        i += 1
        if i == 100000:
            break
print("SUCCESS: Dataset Loaded")
random.shuffle(dataset)
dataset = dataset[:N]

tokenizer_hf = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_CKPT_DIR)
tokenizer_hf.pad_token = "[PAD]"
tokenizer_hf.pad_token_id = 0

# Load the safetensors model
llm = AutoModelForCausalLM.from_pretrained(CKPT_DIR, use_safetensors=True)
if multi_gpu and SINGLE_GPU:
    llm = torch.nn.DataParallel(llm)
llm = llm.to(device)
print(f"{args.model_format} model loaded")

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

if isinstance(llm, nn.DataParallel):
    num_used_gpus = len(llm.device_ids)
    print(f"Number of GPUs used by DataParallel: {num_used_gpus}")

exit(0)

batch = dataset[:B]
input_ids = tokenizer_hf(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
input_ids = input_ids.to(device)
target_ids = input_ids.clone()
target_ids[target_ids == 0] = -100 # set -100 to padding tokens

# Get logits
with torch.no_grad():
    outputs = llm(input_ids)
    logits = outputs.logits
loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])

# Test Generation
print("\n\n\nSTART: Generation+Ranking Test")

start_time = time()

all_ranks = []
for sample_idx in range(0, B):
    ctx = input_ids[sample_idx : sample_idx + 1, :CTX_LEN].to(device)
    ranks = []
    for tok_idx in tqdm(range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN))):
        if input_ids[sample_idx, CTX_LEN + tok_idx] == 0:
            break
        
        with torch.no_grad():
            outputs = llm(ctx)
            logits = outputs.logits
            logits = logits[:, -1, :]

        # Evaluate rank of the ground truth token to the logits
        ground_truth_token_id = input_ids[sample_idx, CTX_LEN + tok_idx].item()

        rank = (
            torch.argsort(logits, descending=True)
            .squeeze()
            .tolist()
            .index(ground_truth_token_id)
        )

        if rank > 500:
            print("Ground Truth Token:", tokenizer_hf.decode([ground_truth_token_id]))
        ranks.append(rank)

        # Use ground truth token to update context
        ctx = torch.cat(
            [
                ctx,
                input_ids[sample_idx : sample_idx + 1, CTX_LEN + tok_idx]
                .unsqueeze(1)
                .to(device),
            ],
            dim=1,
        )

    all_ranks.append(ranks)

rank_avgs = [sum(ranks) / len(ranks) for ranks in all_ranks]
avg_rank = sum(rank_avgs) / len(rank_avgs)

end_time = time() - start_time
print(f"the time for generation test of {args.model_format} model is {end_time}")

