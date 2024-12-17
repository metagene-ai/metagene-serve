import argparse
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
from transformers import PreTrainedTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_dir_4bit", type=str)
    parser.add_argument("--model_dir_8bit", type=str)
    parser.add_argument("--model_ckpt", type=str)
    args = parser.parse_args()

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_dir_4bit = args.model_dir_4bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/gptq-4bit"
    args.model_dir_8bit = args.model_dir_8bit or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/gptq-8bit"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_ckpt}"
    model_dir_4bit = f"{args.model_dir_4bit}/{args.model_ckpt}"
    model_dir_8bit = f"{args.model_dir_8bit}/{args.model_ckpt}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    # TODO: a bigger dataset, maybe the clean tokens
    examples = [
        "AA", "GG", "TAC", "AAAA", "ACCC", "ATCC", "TTCC", "AGCC",
        "ATTTCACCGC",
        "TGCCTCCCGTAGG",
        "TCATTATGCAAAAGGC",
        "GTATTACCGCGGCTGCTGGC",
        "ACTACCAGGGTATCTAATCCTGTT",
        "ACCGTTGCCGGCGTACTCCCCAGGTGGATAGCTTAATGGTTTCCCTCAGGCACCC"
    ]
    examples = [tokenizer(seq, return_tensors="pt").to(torch.device("cuda")) for seq in examples]

    print(f"GPTQ 4bit quantization with {model_dir} ...")
    quantize_config_gptq_4bit = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True)
    model_gptq_4bit = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config_gptq_4bit,
        device_map="cuda")
    model_gptq_4bit.quantize(examples)
    model_gptq_4bit.save_quantized(model_dir_4bit, use_safetensors=True)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"GPTQ quantized model saved to {model_dir_4bit}")

    del model_gptq_4bit

    print(f"8bit quantization with {model_dir} ...")
    quantize_config_gptq_8bit = BaseQuantizeConfig(
        bits=8,
        group_size=128,
        desc_act=True)
    model_gptq_8bit = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config_gptq_8bit,
        device_map="cuda")
    model_gptq_8bit.quantize(examples)
    model_gptq_8bit.save_quantized(model_dir_8bit, use_safetensors=True)
    tokenizer.save_pretrained(model_dir_8bit)
    print(f"GPTQ quantized model saved to {model_dir_8bit}")

    del model_gptq_8bit


if __name__ == "__main__":
    main()