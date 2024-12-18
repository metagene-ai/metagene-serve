import argparse
import random
import time
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_dir_4bit", type=str)
    parser.add_argument("--model_dir_8bit", type=str)
    parser.add_argument("--model_ckpt", type=str)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_dir_4bit = args.model_dir_4bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/bnb-4bit"
    args.model_dir_8bit = args.model_dir_8bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/bnb-8bit"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"
    model_dir_4bit = f"{args.model_dir_4bit}/{args.model_ckpt}"
    model_dir_8bit = f"{args.model_dir_8bit}/{args.model_ckpt}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

    print(f"Bnb 4bit quantization with {model_dir} ...")
    quant_config_bnb_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model_bnb_4bit = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config_bnb_4bit,
        device_map="auto")
    model_bnb_4bit.save_pretrained(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"Bnb 4bit quantized model saved to {model_dir_4bit}")

    del model_bnb_4bit # release memory

    print(f"Bnb 8bit quantization with {model_dir} ...")
    quant_config_bnb_8bit = BitsAndBytesConfig(
        load_in_8bit=True)
    model_bnb_8bit = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config_bnb_8bit,
        device_map="auto")
    model_bnb_8bit.save_pretrained(model_dir_8bit)
    tokenizer.save_pretrained(model_dir_8bit)
    print(f"Bnb 8bit quantized model saved to {model_dir_8bit}")


if __name__ == "__main__":
    main()