import argparse
from awq import AutoAWQForCausalLM
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, AwqConfig
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_dir_4bit", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--data_file", type=str)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_dir_4bit = args.model_dir_4bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/awq-4bit"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    args.data_file = args.data_file or "/project/neiswang_1391/MGFM/MGFM-serving/datasets/evaluate/quantize/perfect_samples.txt"
    return args

# https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY
def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"
    model_dir_4bit = f"{args.model_dir_4bit}/{args.model_ckpt}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calib_data = []
    with open(args.data_file, "r") as f:
        for line in f:
            calib_data.append(line.strip())

    print(f"AWQ 4bit quantization with {model_dir} ...")
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    quantize_config_awq_4bit = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"}
    model.quantize(tokenizer,
                   quant_config=quantize_config_awq_4bit,
                   calib_data=calib_data)
                   # max_calib_seq_len=4)

    quantize_config_awq_4bit_hf = AwqConfig(
        bits=quantize_config_awq_4bit["w_bit"],
        group_size=quantize_config_awq_4bit["q_group_size"],
        zero_point=quantize_config_awq_4bit["zero_point"],
        version=quantize_config_awq_4bit["version"].lower(),
    ).to_dict()
    model.model.config.quantization_config = quantize_config_awq_4bit_hf
    model.save_quantized(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"AWQ 4bit quantized model saved to {model_dir_4bit}")


if __name__ == "__main__":
    main()