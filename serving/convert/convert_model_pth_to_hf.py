import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_model_dir", type=str)
    parser.add_argument("--st_model_dir", type=str)
    parser.add_argument("--model_ckpt", type=str)
    args = parser.parse_args()

    # Set default values
    args.pth_model_dir = args.pth_model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/pth"
    args.st_model_dir = args.st_model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    return args

def main():
    args = parse_args()
    pth_model_dir = f"{args.pth_model_dir}/{args.model_ckpt}"
    st_model_dir = f"{args.st_model_dir}/{args.model_ckpt}"

    pth_model_path = f"{pth_model_dir}/model.pth"
    pth_config_path = f"{pth_model_dir}/config.json"

    print("Load pth model ...")
    config = AutoConfig.from_pretrained(pth_config_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(torch.load(pth_model_path))
    print("pth model loaded")

    model.save_pretrained(st_model_dir)
    print("safetensors model saved")


if __name__ == "__main__":
    main()