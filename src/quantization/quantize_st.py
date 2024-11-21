import random
import torch
import argparse
import time
from transformers import \
    PreTrainedTokenizerFast, \
    AutoModelForCausalLM, \
    GPTQConfig, \
    BitsAndBytesConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization type")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of data for GPTQ quantization")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of original model")
    parser.add_argument("--quant_type", type=str, required=True, help="Quantization type")
    parser.add_argument("--quant_model_dir", type=str, required=True, help="Directory to save quantized model")
    args = parser.parse_args()

    N = 1000  # Dataset clip size

    # Set device to cuda gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0

    start_time = 0
    # Set quantization configs and quantize the model
    if args.quant_type == "gptq":
        print("START: loading original dataset")
        random.seed(42)
        dataset = []
        with open(args.data_dir, "r") as f:
            i = 0
            for line in f:
                dataset.append("_" + line.strip())
                i += 1
                if i == 100000:
                    break
        print("SUCCESS: original dataset loaded")

        # Clip off the first N samples for sanity check
        random.shuffle(dataset)
        quant_dataset = dataset[N:N+10]

        start_time = time.time()
        print(f"START: quantizing the model using {args.quant_type}")
        # GPTQ configs
        gptq_quant_config = GPTQConfig(
            bits=4,
            dataset=quant_dataset,
            tokenizer=tokenizer,
            block_name_to_quantize="model.layers",
        )
        # GPTQ quantized model
        gptq_quantized_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=gptq_quant_config,
            device_map="auto"
        )
        gptq_quantized_model.save_pretrained(args.quant_model_dir)
    elif args.quant_type == "nf4":
        start_time = time.time()
        print(f"START: quantizing the model using {args.quant_type}")
        # NF4 configs
        nf4_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # NF4 quantized model
        nf4_quantized_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=nf4_quant_config,
            device_map="auto"
        )
        nf4_quantized_model.save_pretrained(args.quant_model_dir)

    print(f"SUCCESS: model quantized using {args.quant_type}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Model quantization time: {execution_time} seconds for {args.quant_type} quantization type")
