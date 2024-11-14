import torch
from huggingface_hub import snapshot_download
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
from transformers import \
    AutoModelForCausalLM, \
    AutoConfig, \
    AutoTokenizer, \
    GPTQConfig, \
    BitsAndBytesConfig
from huggingface_hub import login
login(token="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")


def check_model_distribution(model):
    device_ids = set()
    for name, param in model.named_parameters():
        device_id = param.device
        device_ids.add(device_id)
        print(f"Parameter '{name}' is on device {device_id}")
    print(f"\nModel is distributed across devices: {device_ids}")


def check_gpu_usage():
    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3  # GB
        print(f" - GPU {i}: Allocated Memory: {allocated:.2f} GB, Reserved Memory: {reserved:.2f} GB")


def main():
    # Initialize the Accelerator with multi-GPU support
    global accelerator, device
    accelerator = Accelerator(mixed_precision="fp16", device_placement=True)
    device = accelerator.device

    # Load a tokenizer and a model
    print("Loading test model ...")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded")

    import os
    offload_folder = os.getenv('OFFLOAD_FOLDER')
    os.makedirs(offload_folder, exist_ok=True)

    # manually set the device map, using auto or balanced does not work
    # num_layers = getattr(base_model.config, "num_hidden_layers", None)
    # print(f"Number of layers from config: {num_layers}")
    device_map = {str(i): "cuda:0" for i in range(16)}  # Layers 0-15 to cuda:0
    device_map.update({str(i): "cuda:1" for i in range(16, 32)})  # Layers 16-31 to cuda:1

    quantize_config = BnbQuantizationConfig(
        load_in_4bit=True,
        # llm_int8_threshold=6.0
    )
    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
        )

    # for name, module in base_model.named_modules():
    #     if "bitsandbytes" in str(type(module)) and hasattr(module, "to"):
    #         module.to(accelerator.device)  # Ensures each layer initializes on the correct device

    print("Base model loaded")

    # sometimes you have to pre-run theis download before submitting to slurm
    weights_location = snapshot_download(repo_id=model_name)
    print("Weights downloaded")

    # model = model.to(device)
    model = load_and_quantize_model(
        base_model,
        bnb_quantization_config=quantize_config,
        device_map="auto",
        weights_location=weights_location,
        offload_folder=offload_folder
    )
    # quantize_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0 # must be floating-point
    # )
    # model_8bit = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=quantize_config,
    #     # device_map="balanced",
    #     device_map={"0": "cuda:0", "1": "cuda:1"},
    #     offload_folder = offload_folder
    # )
    print("Quantized model loaded")
    print("\n")

    # Move model to Accelerator device and check distribution
    model, tokenizer = accelerator.prepare(model, tokenizer)
    check_model_distribution(model)
    print("\n")

    # Check GPU memory usage before forward pass
    check_gpu_usage()
    print("\n")

    # Run a forward pass to verify GPU utilization
    accelerator.print("Running forward pass on multiple GPUs.")
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    # inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True)
    # inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Model output shape: {outputs.logits.shape}")

    # Check GPU memory usage after forward pass
    check_gpu_usage()
    print("\n")

    print("Test completed successfully on multiple GPUs.")


if __name__ == "__main__":
    main()
