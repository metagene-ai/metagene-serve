import random
import torch
import argparse
import time
from transformers import \
    PreTrainedTokenizerFast, \
    AutoModelForCausalLM, \
    GPTQConfig, \
    BitsAndBytesConfig

DATASET_DIR = "../data/sanity_check/cleaned_tokens_2000000000.txt"
TOKENIZER_CKPT_DIR = "../model_ckpts/litgpt/step-00078000/"
ST_CKPT_DIR = "../model_ckpts/safetensors/step-00078000/"
GPTQ_ST_CKPT_DIR = "../model_ckpts/safetensors/gptq/step-00078000"
NF4_ST_CKPT_DIR = "../model_ckpts/safetensors/nf4/step-00078000"

parser = argparse.ArgumentParser(description="Quantization type")
parser.add_argument("--quant_type", type=str, required=True, help="Quantization type")
args = parser.parse_args()

N = 1000  # Dataset clip size

# Set device to cuda gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_CKPT_DIR)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = 0

start_time = 0
# Set quantization configs and quantize the model
if args.quant_type == "gptq":
    print("START: loading original dataset")
    random.seed(42)
    dataset = []
    with open(DATASET_DIR, "r") as f:
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
        ST_CKPT_DIR,
        quantization_config=gptq_quant_config,
        device_map="auto"
    )
    gptq_quantized_model.save_pretrained(GPTQ_ST_CKPT_DIR)
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
        ST_CKPT_DIR,
        quantization_config=nf4_quant_config,
        device_map="auto"
    )
    nf4_quantized_model.save_pretrained(NF4_ST_CKPT_DIR)

print(f"SUCCESS: model quantized using {args.quant_type}")
end_time = time.time()
execution_time = end_time - start_time
print(f"Model quantization time: {execution_time} seconds for {args.quant_type} quantization type")
