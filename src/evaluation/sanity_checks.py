import random
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, LlamaTokenizer

import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check model format")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of data for sanity check")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory of output")
    parser.add_argument("--model_format", type=str, required=True, help="Model format for sanity check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    N = 1000 # Dataset Size
    B = 32 # Batch Size
    CTX_LEN = 12 # Context Length
    GEN_LEN = 20 # Generation Length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("START: loading dataset")
    random.seed(42)
    dataset = []
    with open(args.data_dir, "r") as f:
        i = 0
        for line in f:
            dataset.append("_" + line.strip())
            i += 1
            if i == 100000:
                break
    print("SUCCESS: dataset loaded")
    random.shuffle(dataset)
    dataset = dataset[:N]

    tokenizer_hf = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    tokenizer_hf.pad_token = "[PAD]"
    tokenizer_hf.pad_token_id = 0

    start_time = time.time()

    if args.model_format == "litgpt":
        print(args.model_dir)
        llm = LLM.load(args.model_dir)
        print("litgpt model loaded")
    elif args.model_format == "st" or args.model_format == "st_gptq" or args.model_format == "st_nf4":
        # Load the safetensors model
        llm = AutoModelForCausalLM.from_pretrained(args.model_dir, use_safetensors=True)
        llm = llm.to(device)
        print("safetensors model loaded")
    else:
        print("Such model for sanity check is not implemented")
        exit(1)


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
            elif args.model_format != "gguf":
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


    # save all ranks, each row is a rank of current sequence
    file_name = f"{args.output_dir}/all_ranks.txt"
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
    print(f"SUCCESS: saving results from generation and ranking test to all_ranks.txt under {args.output_dir}")
