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
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_dir_4bit = args.model_dir_4bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/awq-4bit"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

# https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY
def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"
    model_dir_4bit = f"{args.model_dir_4bit}/{args.model_ckpt}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    # TODO need a bigger dataset with large size than 512?
    examples = [
        "AA", "GG", "TAC", "AAAA", "ACCC", "ATCC", "TTCC", "AGCC",
        "ATTTCACCGC",
        "TGCCTCCCGTAGG",
        "TCATTATGCAAAAGGC",
        "GTATTACCGCGGCTGCTGGC",
        "ACTACCAGGGTATCTAATCCTGTT",
        "ACCGTTGCCGGCGTACTCCCCAGGTGGATAGCTTAATGGTTTCCCTCAGGCACCC"
    ]
    calib_data = examples

    print(f"AWQ 4bit quantization with {model_dir} ...")
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    quantize_config_awq_4bit = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"}
    model.quantize(tokenizer,
                   quant_config=quantize_config_awq_4bit,
                   calib_data=calib_data,
                   max_calib_seq_len=4) # TODO change to the default 512

    quantize_config_awq_4bit_hf = AwqConfig(
        bits=awq_4bit_quant_config["w_bit"],
        group_size=awq_4bit_quant_config["q_group_size"],
        zero_point=awq_4bit_quant_config["zero_point"],
        version=awq_4bit_quant_config["version"].lower(),
    ).to_dict()
    model.model.config.quantization_config = quantize_config_awq_4bit_hf
    model.save_quantized(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"AWQ 4bit quantized model saved to {model_dir_4bit}")


if __name__ == "__main__":
    main()