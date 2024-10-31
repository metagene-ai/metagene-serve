import random
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


parser = argparse.ArgumentParser(description="Sanity check model format")
parser.add_argument("--model_format", type=str, required=True, help="Model format for sanity check")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for sanity check")
args = parser.parse_args()

# TO USE THIS SCRIPT, PLEASE CHANGE THE FOLLOWING DIRECTORIES
if args.model_format == "litgpt":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/step-00078000/"
elif args.model_format == "pth":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_pth/step-00078000/"
elif args.model_format == "st":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000/"
elif args.model_format == "st_gptq":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/gptq_safetensors/step-00078000/"
elif args.model_format == "st_nf4":
    CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_safetensors/nf4_safetensors/step-00078000/"
# elif args.model_format == "gguf":
#     CKPT_DIR = "/workspace/MGFM/model_ckpts/converted_gguf/step-00078000/"

DATASET_DIR = "/workspace/MGFM/data/sanity_check/cleaned_tokens_2000000000.txt"
TOKENIZER_CKPT_DIR = "/workspace/MGFM/model_ckpts/step-00078000/"

N = 1000 # Dataset Size
B = int(args.batch_size) # Batch Size
CTX_LEN = 12 # Context Length
GEN_LEN = 20 # Generation Length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("START: loading dataset")
random.seed(42)
dataset = []
with open(DATASET_DIR, "r") as f:
    i = 0
    for line in f:
        dataset.append("_" + line.strip())
        i += 1
        if i == 100000:
            break
print("SUCCESS: dataset loaded")
random.shuffle(dataset)
dataset = dataset[:N]

tokenizer_hf = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_CKPT_DIR)
tokenizer_hf.pad_token = "[PAD]"
tokenizer_hf.pad_token_id = 0

start_time = time.time()

if args.model_format == "litgpt":
    llm = LLM.load(CKPT_DIR)
    print("litgpt model loaded")
elif args.model_format == "pth":
    # Load the pth model
    config = AutoConfig.from_pretrained(CKPT_DIR + "config.json")
    llm = AutoModelForCausalLM.from_config(config)
    llm.load_state_dict(torch.load(CKPT_DIR + "model.pth"))
    llm = llm.to(device)
    print("pth model loaded")
elif args.model_format == "st" or args.model_format == "st_gptq" or args.model_format == "st_nf4":
    # Load the safetensors model
    llm = AutoModelForCausalLM.from_pretrained(CKPT_DIR, use_safetensors=True)
    llm = llm.to(device)
    print("safetensors model loaded")
# elif args.model_format == "gguf":
#     # Load the gguf model
#     llm = Llama(model_path=CKPT_DIR + "model.gguf")
#     llm = llm.to(device)
#     print("gguf model loaded")

end_time = time.time()
execution_time = end_time - start_time
print(f"Model load time: {execution_time} seconds for {args.model_format} model format")

print('\n\n\nSTART: valiation loss on a batch')
batch = dataset[:B]
input_ids = tokenizer_hf(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
input_ids = input_ids.to(device)
target_ids = input_ids.clone()
target_ids[target_ids == 0] = -100 # set -100 to padding tokens

# Get logits over a batch
if args.model_format == "litgpt":
    logits = llm.model(input_ids)
elif args.model_format != "gguf":
    with torch.no_grad():
        outputs = llm(input_ids)
        logits = outputs.logits
# else:
#     logits = llm.model(input_ids)
#     loss_gguf = chunked_cross_entropy(logits_gguf[..., :-1, :], target_ids[..., 1:])
loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
print("SUCCESS: validation completed")

# print("\n\n\nSAVING logits to sample_logits.txt")
# sliced_logits = logits[..., :-1, :]
# reshaped_logits = sliced_logits.reshape(-1, sliced_logits.shape[-1])
# reshaped_logits_cpu = reshaped_logits.cpu().numpy()
# np.savetxt(f"sample_logits_{args.model_format}.txt", reshaped_logits_cpu, delimiter=',')

# Test Generation and Ranking
print("\n\n\nSTART: Generation and Ranking Test")
all_ranks = []
for sample_idx in range(0, B):
    ctx = input_ids[sample_idx : sample_idx + 1, :CTX_LEN].to(device)
    
    ranks = []
    for tok_idx in tqdm(range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN))):
        if input_ids[sample_idx, CTX_LEN + tok_idx] == 0:
            break
        
        if args.model_format == "litgpt":
            logits = llm.model(ctx)[:, -1, :]
        if args.model_format != "gguf":
            with torch.no_grad():
                outputs = llm(ctx)
                logits = outputs.logits
                logits = logits[:, -1, :]
        # else:
        #     logits = llm.model(ctx)[:, -1, :]

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
print("SUCCESS: saving results from generation and ranking test to all_ranks.txt")

# save all ranks, each row is a rank of current sequence
file_name = f"all_ranks_{args.model_format}.txt"
with open(file_name, "w") as f:
    f.write(f"Validation loss: {loss.item()}\n")
    f.write(f"Average rank: {avg_rank}\n")
    f.write(f"Rank averages:\n")
    f.write(f"[")
    for rank_avg in rank_avgs:
        f.write(f'{rank_avg}')
    f.write(f"]\n")
    for ranks in all_ranks:
        f.write(",".join(map(str, ranks)) + "\n")
