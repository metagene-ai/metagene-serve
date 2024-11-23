from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
import os
import pytest
import random
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


@pytest.fixture
def model_format(request):
    return request.config.getoption("model_format")

@pytest.fixture
def model_dir(request):
    return request.config.getoption("model_dir")

@pytest.fixture
def data_path(request):
    return request.config.getoption("data_path")

@pytest.fixture
def output_dir(request):
    output_dir = request.config.getoption("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.fixture
def dataset(data_path):
    N = 1000
    random.seed(42)
    dataset = []
    with open(data_path, "r") as f:
        i = 0
        for line in f:
            dataset.append("_" + line.strip())
            i += 1
            if i == 100000:
                break
    random.shuffle(dataset)
    dataset = dataset[:N]
    return dataset

@pytest.fixture
def tokenizer(model_dir):
    tokenizer_hf = PreTrainedTokenizerFast.from_pretrained(model_dir)
    tokenizer_hf.pad_token = "[PAD]"
    tokenizer_hf.pad_token_id = 0
    return tokenizer_hf

@pytest.fixture
def model(model_format, model_dir):
    if model_format == "litgpt":
        model = LLM.load(model_dir)
    elif model_format in {"st", "st_gptq", "st_nf4"}:
        model = AutoModelForCausalLM.from_pretrained(model_dir, use_safetensors=True).to('cuda')
    else:
        raise NotImplementedError(f"Model format {model_format} is not implemented.")
    return model


def test_validation(model_format, output_dir, dataset, tokenizer, model):
    B = 32  # Batch Size
    batch = dataset[:B]
    input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
    input_ids = input_ids.to('cuda')
    target_ids = input_ids.clone()
    target_ids[target_ids == 0] = -100  # set -100 to padding tokens

    if model_format == "litgpt":
        logits = model.model(input_ids)
    elif model_format != "gguf":
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
    else:
        raise NotImplementedError(f"Model format {model_format} is not implemented.")

    loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
    assert loss.item() < 10, f"Validation loss is too high: {loss.item()}"

    output_file = f"{output_dir}/ret_val.txt"
    with open(output_file, "w") as f:
        f.write(f"Validation loss: {loss.item()}\n")
    print(f"Validation results saved to {output_file}")

def test_generation(output_dir, dataset, tokenizer, model):
    B = 32  # Batch Size
    CTX_LEN = 12  # Context Length
    GEN_LEN = 20  # Generation Length
    batch = dataset[:B]
    input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
    input_ids = input_ids.to('cuda')

    all_ranks = []
    for sample_idx in range(0, B):
        ctx = input_ids[sample_idx : sample_idx + 1, :CTX_LEN].to('cuda')

        ranks = []
        for tok_idx in range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN)):
            if input_ids[sample_idx, CTX_LEN + tok_idx] == 0:
                break

            if model_format == "litgpt":
                logits = model.model(ctx)[:, -1, :]
            elif model_format != "gguf":
                with torch.no_grad():
                    outputs = model(ctx)
                    logits = outputs.logits[:, -1, :]
            else:
                raise NotImplementedError(f"Model format {model_format} is not implemented.")

            ground_truth_token_id = input_ids[sample_idx, CTX_LEN + tok_idx].item()
            rank = (
                torch.argsort(logits, descending=True)
                .squeeze()
                .tolist()
                .index(ground_truth_token_id)
            )
            ranks.append(rank)

            # Use ground truth token to update context
            ctx = torch.cat(
                [ctx, input_ids[sample_idx : sample_idx + 1, CTX_LEN + tok_idx].unsqueeze(1)],
                dim=1,
            )
        all_ranks.append(ranks)

    rank_avgs = [sum(ranks) / len(ranks) for ranks in all_ranks]
    avg_rank = sum(rank_avgs) / len(rank_avgs)
    assert avg_rank < 500, f"Average rank is too high: {avg_rank}"

    # Save all ranks to a file
    output_file = f"{output_dir}/ret_gen.txt"
    with open(output_file, "w") as f:
        f.write(f"Average rank: {avg_rank}\n")
        f.write(f"Rank averages:\n")
        f.write(f"[")
        for rank_avg in rank_avgs:
            f.write(f'{rank_avg}')
        f.write(f"]\n")
        for ranks in all_ranks:
            f.write(",".join(map(str, ranks)) + "\n")
    print(f"Generation results saved to {output_file}")
