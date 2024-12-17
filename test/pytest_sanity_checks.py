from litgpt import LLM
from litgpt.utils import chunked_cross_entropy
from llama_cpp import Llama
import os
import pytest
import random
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


@pytest.fixture
def model_format(request):
    return request.config.getoption("sanity_check_model_format")

@pytest.fixture
def model_dir(request):
    return request.config.getoption("sanity_check_model_dir")

@pytest.fixture
def data_path(request):
    return request.config.getoption("sanity_check_data_path")

@pytest.fixture
def output_dir(request):
    output_dir = request.config.getoption("sanity_check_output_dir")
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
        model = Llama(model_path=f"{model_dir}/model.gguf")
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
        outputs = model(input_ids)
        logits = outputs.get("logits", None)

    loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
    assert loss.item() < 10, f"Validation loss is too high: {loss.item()}"

    output_file = f"{output_dir}/ret_val.txt"
    with open(output_file, "w") as f:
        f.write(f"Validation loss: {loss.item()}\n")
    print(f"Validation results saved to {output_file}")

def test_generation(output_dir, dataset, tokenizer, model, model_format):
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


import torch


def test_generation(output_dir, dataset, tokenizer, model, model_format):
    B = 32  # Batch Size
    CTX_LEN = 12  # Context Length
    GEN_LEN = 20  # Generation Length
    SANITY_GEN_LEN = 50  # Length for autoregressive generation in sanity check
    batch = dataset[:B]
    input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
    input_ids = input_ids.to('cuda')

    all_ranks = []
    perfect_samples = []  # To track samples with all ranks == 0

    for sample_idx in range(B):
        # Initialize context with the first CTX_LEN tokens
        ctx = input_ids[sample_idx: sample_idx + 1, :CTX_LEN].clone().to('cuda')

        ranks = []
        is_perfect = True  # Flag to check if all ranks are 0
        use_autoregression = False  # Flag to indicate if autoregression should be used

        for tok_idx in range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN)):
            current_token_id = input_ids[sample_idx, CTX_LEN + tok_idx].item()

            # Break if the current token is a padding token
            if current_token_id == 0:
                break

            # Generate logits based on the current context
            if model_format == "litgpt":
                logits = model.model(ctx)[:, -1, :]
            elif model_format != "gguf":
                with torch.no_grad():
                    outputs = model(ctx)
                    logits = outputs.logits[:, -1, :]
            else:
                raise NotImplementedError(f"Model format {model_format} is not implemented.")

            # Compute the rank of the ground truth token
            ground_truth_token_id = current_token_id
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_indices = sorted_indices.squeeze().tolist()
            if isinstance(sorted_indices, int):
                sorted_indices = [sorted_indices]
            try:
                rank = sorted_indices.index(ground_truth_token_id)
            except ValueError:
                # If the ground truth token is not in the sorted indices
                rank = len(sorted_indices)
            ranks.append(rank)

            # Determine whether to switch to autoregression
            if rank == 0:
                use_autoregression = True
            else:
                is_perfect = False  # Found a token with rank > 0

            # Update the context based on the autoregression flag
            if use_autoregression:
                # Autoregressive generation: use the model's top prediction
                predicted_token_id = sorted_indices[0]
                ctx = torch.cat(
                    [ctx, torch.tensor([[predicted_token_id]], device='cuda')],
                    dim=1
                )
            else:
                # Use the ground truth token
                ctx = torch.cat(
                    [ctx, input_ids[sample_idx: sample_idx + 1, CTX_LEN + tok_idx].unsqueeze(1)],
                    dim=1
                )

        all_ranks.append(ranks)

        if is_perfect:
            # Store the initial context and the full input_ids for perfect samples
            perfect_samples.append((sample_idx, input_ids[sample_idx]))

    # Calculate average ranks
    rank_avgs = [sum(ranks) / len(ranks) for ranks in all_ranks if len(ranks) > 0]
    avg_rank = sum(rank_avgs) / len(rank_avgs) if rank_avgs else float('inf')

    # Assertion to ensure average rank is below the threshold
    assert avg_rank < 500, f"Average rank is too high: {avg_rank}"

    # ----------------------------
    # Sanity Check: Autoregressive Generation on Perfect Samples
    # ----------------------------
    if perfect_samples:
        print(f"Performing autoregressive generation on {len(perfect_samples)} perfectly predicted samples.")

        # Prepare inputs for generation
        perfect_contexts = [sample[1][:CTX_LEN] for sample in perfect_samples]
        perfect_contexts = torch.stack(perfect_contexts).to('cuda')

        # Decode the initial context for logging or further processing
        initial_texts = tokenizer.batch_decode(perfect_contexts, skip_special_tokens=True)

        # Configure generation parameters for deterministic generation
        generation_params = {
            "max_length": CTX_LEN + SANITY_GEN_LEN,
            "temperature": 0.0,  # Ensures deterministic generation
            "do_sample": False,  # Disables sampling to make generation deterministic
            "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        }

        # Perform generation
        with torch.no_grad():
            generated_outputs = model.generate(
                perfect_contexts,
                **generation_params
            )

        # Decode the generated outputs
        generated_texts = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        for idx, (initial, generated) in enumerate(zip(initial_texts, generated_texts)):
            print(f"\nSample {idx + 1} - Initial Context:\n{initial}")
            print(f"Sample {idx + 1} - Generated Text:\n{generated}")

        with open(f"{output_dir}/new_sanity_generated.txt", "w") as f:
            for text in generated_texts:
                f.write(text + "\n")

    else:
        print("No perfectly predicted samples found for sanity check.")
