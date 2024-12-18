# https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing#scrollTo=U8-r_YXPGf6G
# https://github.com/huggingface/optimum-quanto
# https://github.com/huggingface/optimum-quanto/issues/136
import argparse
from optimum.quanto import QuantizedModelForCausalLM, qint4, qint8, freeze, quantize
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_dir_8bit", type=str)
    parser.add_argument("--model_ckpt", type=str)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_dir_8bit = args.model_dir_8bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/quanto-8bit"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"
    model_dir_4bit = f"{args.model_dir_4bit}/{args.model_ckpt}"
    model_dir_8bit = f"{args.model_dir_8bit}/{args.model_ckpt}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

    print(f"Quanto 4bit quantization with {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    model_quanto_4bit = QuantizedModelForCausalLM.quantize(
        model,
        weights=qint4,
        exclude='lm_head')
    freeze(model_quanto_4bit)
    model_quanto_4bit.save_pretrained(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"Quanto quantized model saved to {model_dir_4bit}")


if __name__ == "__main__":
    main()