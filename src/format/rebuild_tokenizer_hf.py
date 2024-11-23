import argparse
import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    model_path = args.model_path or "/project/MGFM/MGFM-serving/model_ckpts/safetensors/step-00078000"

    tokenizer_backup_path = f"{model_path}/tokenizer_backup.json"
    tokenizer_path = f"{model_path}/tokenizer.json"
    if not os.path.exists(tokenizer_backup_path):
        if os.path.exists(tokenizer_path):
            os.rename(tokenizer_path, tokenizer_backup_path)
        else:
            print("No tokenizer.json available!")
            exit(1)

    with open(tokenizer_backup_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # generate vocab.json and merges.txt
    print("Generating vocab merges ...")
    vocab_file = f"{model_path}/vocab.json"
    merges_file = f"{model_path}/merges.txt"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data["model"]["vocab"], f, indent=2, ensure_ascii=False)
    with open(merges_file, 'w', encoding='utf-8') as f:
        for merge in tokenizer_data["model"]["merges"]:
            f.write(merge + "\n")

    # Create a BPE tokenizer
    print("Setting pre_tokenizer and decoder ...")
    tokenizer_rebuilt = Tokenizer(models.BPE.from_file(vocab_file, merges_file))
    # Set pre_tokenizer
    tokenizer_rebuilt.pre_tokenizer = pre_tokenizers.Whitespace()
    # Set decoder
    tokenizer_rebuilt.decoder = decoders.ByteLevel()
    tokenizer_rebuilt.save(tokenizer_path)

    # Set post_processor
    print("Setting post_processor ...")
    with open(tokenizer_backup_path, 'r') as source_file:
        source_data = json.load(source_file)
    with open(tokenizer_path, 'r') as target_file:
        target_data = json.load(target_file)
    target_data["post_processor"] = source_data["post_processor"]

    # change from null to "[UNK]" after generation
    target_data["model"]["unk_token"] = source_data["model"]["unk_token"]

    with open(tokenizer_path, 'w') as target_file:
        json.dump(target_data, target_file, indent=2, ensure_ascii=False)
