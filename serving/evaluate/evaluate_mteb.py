import argparse
import json
import os
import sys
sys.path.append("/home1/shangsha/workspace/MGFM/MGFM-serving/serving/evaluate/gene-mteb")
import mteb
from tqdm import tqdm
from transformers.trainer_utils import set_seed
import torch
import torch.distributed as dist
from vllm import ModelRegistry

from serving.evaluate.mteb_wrapper import LlamaWrapper, DNABERTWrapper, NTWrapper
from serving.evaluate.vllm_wrapper import vllmLlamaWrapper


def rename_config_model_arch(config_file, name):
    with open(config_file, 'r') as file:
        data = json.load(file)
    data["architectures"] = [name]
    with open(config_file, 'w') as file:
        json.dump(data, file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", nargs='+', required=True)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_dtype", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    # DNABERT:
        # DNABERT-2-117M: "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/baselines/dnabert/DNABERT-2-117M"
        # DNABERT-S: "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/baselines/dnabert/DNABERT-S"
    # safetensors: "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/step-00086000"
    # NT: "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/baselines/nt/nucleotide-transformer-2.5b-multi-species"

    args.model_type = args.model_type or "DNABERT-2-117M"
    args.model_dtype = args.model_dtype or "f32"
    if args.model_dtype == "f32":
        args.model_dtype = torch.float32
    elif args.model_dtype == "f16":
        args.model_dtype = torch.float16
    elif args.model_dtype == "bf16":
        args.model_dtype = torch.bfloat16

    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts"
    args.output_dir = args.output_dir or "/project/neiswang_1391/MGFM/MGFM-serving/outputs/evaluate/mteb"
    args.model_ckpt = args.model_ckpt or "step-00086000" # only for MGFM
    args.seed = args.seed or 42

    if isinstance(args.task_type, str):
        args.task_type = [args.task_type]

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.use_vllm:
        assert args.model_type == "safetensors", "VLLM only supports MGFM models"

        model_dir = f"{args.model_dir}/{args.model_type}/{args.model_ckpt}"
        output_dir = f"{args.output_dir}/vllm/{args.model_ckpt}"

        print("Wrapping vllm model with vllmLlamaWrapper ...")
        model = vllmLlamaWrapper(
            model_dir=model_dir,
            seed=args.seed,
            dtype=args.model_dtype)
    else:
        print("Wrapping model with mtebWrapper ...")
        if args.model_type in [
            "DNABERT-2-117M",
            "DNABERT-S"
        ]:
            model_dir = f"{args.model_dir}/baselines/{args.model_type}"
            output_dir = f"{args.output_dir}/dnabert/{args.model_type}"
            model = DNABERTWrapper(model_dir, args.seed)
        elif args.model_type in [
            "NT-2.5b-multi-species",
            "NT-2.5b-1000g",
            "NT-500m-1000g",
            "NT-500m-human-ref",
            "NT-v2-100m-multi-species",
            "NT-v2-250m-multi-species",
            "NT-v2-500m-multi-species",
            "NT-v2-50m-multi-species"
        ]:
            model_dir = f"{args.model_dir}/baselines/{args.model_type}"
            output_dir = f"{args.output_dir}/nt/{args.model_type}"
            model = NTWrapper(model_dir, args.seed)
        elif args.model_type == "safetensors":
            model_dir = f"{args.model_dir}/safetensors/{args.model_ckpt}"
            output_dir = f"{args.output_dir}/non_vllm/{args.model_ckpt}"
            model = LlamaWrapper(model_dir, args.model_dtype, args.seed)
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")

    print(f"Running mteb tasks with {args.model_type} ...")
    tasks = mteb.get_tasks(tasks=args.task_type)
    evaluation = mteb.MTEB(tasks=tasks)
    print("Running evaluation ...")
    results = evaluation.run(model,
                             overwrite_results=True,
                             output_folder=output_dir,
                             encode_kwargs={"batch_size": 32})

    for mteb_results in results:
        print(f"{args.model_type} on {mteb_results.task_name}: {mteb_results.get_score()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure the process group is destroyed upon exiting
        if dist.is_initialized():
            dist.destroy_process_group()