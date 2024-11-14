import os
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
import torch.distributed as dist


# Ensure CUDA synchronization
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# def init_distributed():
#     dist.init_process_group("nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def print_all_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"{num_gpus} GPUs are being used:")
        for i in range(num_gpus):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0)}")

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
    # init_distributed()
    print_all_gpus()
    print("\n")

    # Load tokenizer and model
    print("Loading test model ...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = GPTQConfig(
        bits=4,
        dataset="wikitext2",
        tokenizer=tokenizer,
    )
    nf4_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
    )
    print("Test model loaded")
    print("\n")

    # DeepSpeed configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 5e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8
            }
        },
        "zero_optimization": {
            "stage": 2
        },
        "gradient_clipping": 1.0,
    }

    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        dist_init_required=True
    )
    check_model_distribution(model)
    print("\n")

    # Check GPU memory usage
    check_gpu_usage()
    print("\n")

    # Forward pass for verification
    print("Running forward pass on multiple GPUs.")
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Model output shape: {outputs.logits.shape}")

    # Check GPU memory usage after forward pass
    check_gpu_usage()
    print("\n")
    print("Test completed successfully on multiple GPUs.")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
