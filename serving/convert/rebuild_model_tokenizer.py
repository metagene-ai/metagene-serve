import argparse
import json
import os
from tokenizers import Tokenizer, Regex
from tokenizers.normalizers import Replace
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_ckpt", type=str)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    print("Tokenizer loaded.")

    # Sync the tokenizer and model values
    tokenizer_config = {
        "do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": tokenizer.cls_token, # stay with None as the model does
        "mask_token": "[MASK]",
        # change from 1000000000000000019884624838656 to 512 (model.config.max_position_embeddings)
        # not changing would trigger cuda device-side assert error
        "model_max_length": 512,
        "tokenizer_class": tokenizer.__class__.__name__
    }

    # Save to the tokenizer_config.json
    with open(f"{model_dir}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=4)
    print("Tokenizer values saved to tokenizer_config.json.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Tokenizer_config.json and tokenizer.json aligned.")

    model_values = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "mask_token": "[MASK]",
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]"
    }
    with open(f'{model_dir}/special_tokens_map.json', 'w') as f:
        json.dump(model_values, f, indent=4)
    print("Model and tokenizer values aligned.")

    # auto _ prepend
    tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    tokenizer.normalizer = Replace(Regex(r"^"), "_")
    print("Auto underscore prepend added.")

    # Update the post-processor to remove [BOS]
    # Example from https://github.com/huggingface/tokenizers/blob/24d29f498d890638279b0d51e899b6020571719d/bindings/python/tests/bindings/test_tokenizer.py#L605
    tokenizer.post_processor = TemplateProcessing(
        single=["$0", "[EOS]"],
        pair=["$A", "$B:1"],
        special_tokens=[
            ("[EOS]", 4),
            ("[MASK]", 5),
            ("[PAD]", 0),
            ("[SEP]", 2)
        ]
    )
    tokenizer.save(f"{model_dir}/tokenizer.json")
    print("[BOS] post-tok removed.")

    # minor adjustment to tokenizer.json
    with open(f"{model_dir}/tokenizer_original.json", 'r') as src_file:
        src_data = json.load(src_file)
    with open(f"{model_dir}/tokenizer.json", 'r') as dest_file:
        dest_data = json.load(dest_file)
    dest_data["post_processor"] = src_data["post_processor"]
    dest_data["model"]["unk_token"] = src_data["model"]["unk_token"]
    dest_data["model"]["merges"] = src_data["model"]["merges"]
    with open(f"{model_dir}/tokenizer.json", 'w') as dest_file:
        json.dump(dest_data, dest_file, indent=2, ensure_ascii=False)

    print("Tokenizer is rebuilt now.")


if __name__ == '__main__':
    main()