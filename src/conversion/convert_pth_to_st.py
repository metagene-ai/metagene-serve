from transformers import AutoConfig, AutoModelForCausalLM
import torch
import argparse


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Convert model format")
    parser.add_argument("--pth_model_dir", type=str, required=True, help="Directory containing model.pth and config.json")
    parser.add_argument("--st_model_dir", type=str, required=True, help="Directory to save safetensors model")
    args = parser.parse_args()

    # Set the paths to your files
    model_path = args.pth_model_dir + "/" + "model.pth"
    config_path = args.pth_model_dir + "/" + "config.json"

    # Load the pth model
    print("loading pth model ...")
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(torch.load(model_path))
    print("pth model loaded")

    # Save the model in safetensors format
    model.save_pretrained(args.st_model_dir)
    print("safetensors model saved")
