from datasets import Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast
from transformers.trainer_utils import set_seed
from transformers.pipelines.pt_utils import KeyDataset
import torch
from tqdm import tqdm
import random


def generate_metgaene_seqs(num_sequences, max_length=20):
    def generate_metgaene_seq(length):
        nucleotides = ['A', 'T', 'C', 'G']
        return ''.join(random.choice(nucleotides) for _ in range(length))

    sequences = []
    for _ in range(num_sequences):
        length = random.randint(1, max_length)
        sequences.append(generate_metgaene_seq(length))
    return sequences


def main():
    set_seed(42)

    model_dir = "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/step-00086000"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto")

    tokenizer_dir = "./test/tokenizer"
    tokenizer_names = [
        "tokenizer", # rebuilt [no], prepend [no], bos [no], eos [no]
        "tokenizer_bos", # rebuilt [no], prepend [no], bos [yes], eos [no]
        "tokenizer_eos",  # rebuilt [no], prepend [no], bos [no], eos [yes]
        "tokenizer_bos_eos", # rebuilt [no], prepend [no], bos [yes], eos [yes]
        "tokenizer_rebuilt",  # rebuilt [yes], prepend [no], bos [no], eos [no]
        "tokenizer_rebuilt_bos",  # rebuilt [yes], prepend [no], bos [yes], eos [no]
        "tokenizer_rebuilt_eos",  # rebuilt [yes], prepend [no], bos [no], eos [yes]
        "tokenizer_rebuilt_bos_eos",  # rebuilt [yes], prepend [no], bos [yes], eos [yes]
        "tokenizer_prepend", # rebuilt [no], prepend [yes], bos [no], eos [no]
        "tokenizer_prepend_bos", # rebuilt [no], prepend [yes], bos [yes], eos [no]
        "tokenizer_prepend_eos",  # rebuilt [no], prepend [yes], bos [no], eos [yes]
        "tokenizer_prepend_bos_eos", # rebuilt [no], prepend [yes], bos [yes], eos [yes]
        "tokenizer_rebuilt_prepend", # rebuilt [yes], prepend [yes], bos [no], eos [no]
        "tokenizer_rebuilt_prepend_bos", # rebuilt [yes], prepend [yes], bos [yes], eos [no]
        "tokenizer_rebuilt_prepend_eos",  # rebuilt [yes], prepend [yes], bos [no], eos [yes]
        "tokenizer_rebuilt_prepend_bos_eos", # rebuilt [yes], prepend [yes], bos [yes], eos [yes]
    ]

    num_sequences = 1000
    max_length = 20
    fail_counts = []
    sequences = generate_metgaene_seqs(num_sequences, max_length)

    for tokenizer_name in tqdm(tokenizer_names):
        tokenizer_file_prefix = f"{tokenizer_dir}/{tokenizer_name}"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_file_prefix}.json")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=32,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto")
        pipeline.tokenizer.pad_token_id = model.config.eos_token_id

        dataset = Dataset.from_dict({"text": sequences})
        outputs = pipeline(
                KeyDataset(dataset, "text"),
                max_new_tokens=42,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True)
        fail_count = sum(
            1
            for output in outputs
            if len([gen for gen in output[0]["generated_text"].split()[1:] if gen not in ["_", "[PAD]", "[EOS]", "[BOS]"]]) == 0
        )
        fail_counts.append(fail_count)

    print("=== Failure Percentage ===")
    for tokenizer_name, fail_count in zip(tokenizer_names, fail_counts):
        print(f"{tokenizer_name}: {100 * fail_count / num_sequences}%")


if __name__ == "__main__":
    main()