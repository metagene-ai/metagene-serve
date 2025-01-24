from awq import AutoAWQForCausalLM
from transformers import PreTrainedTokenizerFast, AwqConfig
from transformers.trainer_utils import set_seed


def main():
    set_seed(42)

    model_name_or_path = "metagene-ai/METAGENE-1"
    model_dir_4bit = "./quantized_models/awq-4bit"

    calib_data = []
    with open("./assets/data/sample_reads.csv", "r") as f:
        for line in f:
            calib_data.append(line.strip())

    print(f"Running AWQ 4bit quantization with ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
    model = AutoAWQForCausalLM.from_pretrained(model_name_or_path)
    quantize_config_awq_4bit = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"}
    model.quantize(tokenizer,
                   quant_config=quantize_config_awq_4bit,
                   calib_data=calib_data,
                   max_calib_seq_len=4)

    quantize_config_awq_4bit_hf = AwqConfig(
        bits=quantize_config_awq_4bit["w_bit"],
        group_size=quantize_config_awq_4bit["q_group_size"],
        zero_point=quantize_config_awq_4bit["zero_point"],
        version=quantize_config_awq_4bit["version"].lower()).to_dict()
    model.model.config.quantization_config = quantize_config_awq_4bit_hf
    model.save_quantized(model_dir_4bit)
    tokenizer.save_pretrained(model_dir_4bit)
    print(f"AWQ 4bit quantized model saved to {model_dir_4bit}")


if __name__ == "__main__":
    main()