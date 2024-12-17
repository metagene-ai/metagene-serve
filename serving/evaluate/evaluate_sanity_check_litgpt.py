import argparse
from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
import numpy as np
import random
import time
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import PreTrainedTokenizerFast
import torch


N = 1000  # dataset size
B = 32  # batch size
CTX_LEN = 12  # context length
GEN_LEN = 20  # generation length
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_prepend", type=int, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts"
    args.model_type = args.model_type or "litgpt"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    args.data_file = args.data_file or "/project/neiswang_1391/MGFM/MGFM-serving/datasets/evaluate/sanity_check/cleaned_tokens_2000000000.txt"
    args.output_dir = args.output_dir or "/project/neiswang_1391/MGFM/MGFM-serving/outputs/evaluate/sanity_check"
    args.seed = args.seed or 42
    return args

def get_dataset(data_file, use_prepend=1):
    if use_prepend:
        print("\nUsing prepend for dataset\n")
    else:
        print("\nNot using prepend for dataset\n")
    dataset = []
    with open(data_file, "r") as f:
        i = 0
        for line in f:
            if use_prepend:
                dataset.append("_" + line.strip())
            else:
                dataset.append(line.strip())
            i += 1
            if i == 100000:
                break
    random.shuffle(dataset)
    dataset = dataset[:N]
    return dataset

def get_model(model_dir, mode_type):
    if mode_type == "litgpt":
        model = LLM.load(model_dir)
    else:
        raise ValueError(f"Invalid model type: {mode_type}")
    return model

def sanity_check_batch_loss(model, model_type, input_ids):
    target_ids = input_ids.clone()
    target_ids[target_ids == 0] = -100

    if model_type == "litgpt":
        logits = model.model(input_ids)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    batch_loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
    return batch_loss

def sanity_check_generation(model, model_type, input_ids, use_gt=True):
    all_ranks = []
    perfect_samples = []

    for sample_idx in tqdm(range(0, B)):
        ctx = input_ids[sample_idx: sample_idx + 1, :CTX_LEN]

        ranks = []
        is_perfect = True  # Flag to check if all ranks are 0

        for tok_idx in range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN)):
            gt_tok = input_ids[sample_idx, CTX_LEN + tok_idx]

            # Break if the current token is a padding token
            if gt_tok == 0:
                break

            if model_type == "litgpt":
                logits = model.model(ctx)[:, -1, :]
            else:
                raise ValueError(f"Invalid model format: {model_type}")

            # Evaluate rank of the ground truth token to the logits

            rank = torch.argsort(logits, descending=True).squeeze().tolist().index(gt_tok.item())
            ranks.append(rank)

            # Use ground truth token to update context
            if use_gt:
                ctx = torch.cat([
                        ctx,
                        input_ids[sample_idx: sample_idx + 1, CTX_LEN + tok_idx].unsqueeze(1)], dim=1)

            # Use generated token to update context
            else:
                if rank == 0:
                    use_autoregression = True
                else:
                    use_autoregression = False
                    is_perfect = False

                if use_autoregression:
                    generated_token_id = torch.argmax(logits).item()
                    ctx = torch.cat([
                            ctx,
                            torch.tensor([[generated_token_id]]).to(DEVICE)], dim=1)
                else:
                    break

        all_ranks.append(ranks)

        if not use_gt:
            if is_perfect:
                # Store the initial context and the full input_ids for perfect samples
                perfect_samples.append((sample_idx, input_ids[sample_idx]))

    rank_avgs = [sum(ranks) / len(ranks) for ranks in all_ranks]
    avg_rank = sum(rank_avgs) / len(rank_avgs)

    if use_gt:
        return avg_rank, rank_avgs, all_ranks
    else:
        return perfect_samples

def main():
    args = parse_args()

    model_dir = f"{args.model_dir}/{args.model_type}/{args.model_ckpt}"
    output_dir = f"{args.output_dir}/{args.model_ckpt}"
    set_seed(args.seed)

    print(f"Running sanity check on model: {model_dir} ...")

    # prepare sample batch data
    dataset = get_dataset(args.data_file, use_prepend=args.use_prepend)
    sample_batch = dataset[:B]

    model = get_model(model_dir, args.model_type)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0

    # remove leading token
    # for tokenizer_rebuilt.json => remove some real tokens
    # for tokenizer_rebuilt_bos.json => remove [BOS]
    # for tokenizer_rebuilt_prepend.json => remove _
    # for tokenizer_rebuilt_prepend_bos.json => remove [BOS] and still have _
    input_ids = tokenizer(sample_batch,
                          return_tensors="pt",
                          padding=True).input_ids[:, 1:].to(DEVICE)

    print("Checking batch loss on model ... ")
    batch_loss = sanity_check_batch_loss(model, args.model_type, input_ids)

    print("Checking ground truth-guided generation on model ... ")
    avg_rank, rank_avgs, all_ranks = sanity_check_generation(
        model, args.model_type, input_ids,
        use_gt=True)

    print("Checking perfect samples on model ... ")
    perfect_samples = sanity_check_generation(
        model, args.model_type, input_ids,
        use_gt=False)
    perfect_samples_indices = [sample_idx for sample_idx, _ in perfect_samples]

    print(f"Batch loss: {batch_loss.item()}")
    print(f"Average rank (ground truth-guided): {avg_rank}")
    print(f"Perfect samples: {perfect_samples_indices}")

    with open(f"{output_dir}/sanity_check_rets.txt", "w") as f:
        f.write(f"Validation loss on sample batch: {batch_loss.item()}\n")

        f.write(f"Average rank (ground truth-guided): {avg_rank}\n")
        f.write(f"All ranks (ground truth-guided):\n")
        for ranks in all_ranks:
            f.write(",".join(map(str, ranks)) + "\n")

    with open(f"{output_dir}/perfect_samples.txt", "w") as f:
        # Collect all sample_idx values in a list and join them as a single line
        f.write(f"perfect sample indices:\n")
        f.write(" ".join(map(str, perfect_samples_indices)) + "\n")
        for sample_idx, sample in perfect_samples:
            f.write(f"Sample {sample_idx}:\n")
            f.write(sample_batch[sample_idx] + "\n")
            f.write(tokenizer.decode(sample) + "\n")


if __name__ == "__main__":
    main()