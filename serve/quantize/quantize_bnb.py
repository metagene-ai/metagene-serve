import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import set_seed


def main():
    set_seed(42)

    model_name_or_path = "metagene-ai/METAGENE-1"
    model_dir_4bit = "./quantized_models/bnb-4bit"
    model_dir_8bit = "./quantized_models/bnb-8bit"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)

    print(f"Running bnb 4bit quantization ...")
    quant_config_bnb_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model_bnb_4bit = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config_bnb_4bit,
        device_map="auto")
    model_bnb_4bit.save_pretrained(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"Bnb 4bit quantized model saved to {model_dir_4bit}")

    del model_bnb_4bit # release memory

    print(f"Running bnb 8bit quantization ...")
    quant_config_bnb_8bit = BitsAndBytesConfig(
        load_in_8bit=True)
    model_bnb_8bit = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config_bnb_8bit,
        device_map="auto")
    model_bnb_8bit.save_pretrained(model_dir_8bit)
    tokenizer.save_pretrained(model_dir_8bit)
    print(f"Bnb 8bit quantized model saved to {model_dir_8bit}")


if __name__ == "__main__":
    main()