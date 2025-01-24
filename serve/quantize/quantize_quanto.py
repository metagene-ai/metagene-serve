from optimum.quanto import QuantizedModelForCausalLM, qint4, freeze
import os
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.trainer_utils import set_seed


def main():
    set_seed(42)

    model_name_or_path = "metagene-ai/METAGENE-1"
    model_dir_4bit = "./quantized_models/quanto-4bit"
    os.mkdir(model_dir_4bit, exist_ok=True)

    print(f"Running quanto 4bit quantization ...")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
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