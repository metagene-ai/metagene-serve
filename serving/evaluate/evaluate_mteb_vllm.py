import argparse
import json

import os
import sys
sys.path.append("/home1/shangsha/workspace/MGFM/MGFM-serving/serving/evaluate/mteb")
import mteb

import torch
import torch.distributed as dist
from vllm import ModelRegistry

from serving.evaluate.utils import vllmLlamaEmbeddingModel, vllmLlamaWrapper


def rename_config_model_arch(config_file, name):
    with open(config_file, 'r') as file:
        data = json.load(file)
    data["architectures"] = [name]
    with open(config_file, 'w') as file:
        json.dump(data, file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", nargs='+', required=True)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    args.model_type = args.model_type or "safetensors"
    args.model_ckpt = args.model_ckpt or "step-00086000"
    args.model_dir = args.model_dir or "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts"
    args.output_dir = args.output_dir or "/project/neiswang_1391/MGFM/MGFM-serving/outputs/evaluate/mteb"
    args.seed = args.seed or 42

    if isinstance(args.task_type, str):
        args.task_type = [args.task_type]

    return args

def main():
    args = parse_args()
    model_dir = f"{args.model_dir}/{args.model_type}/{args.model_ckpt}"
    output_dir = f"{args.output_dir}/vllm/{args.model_ckpt}"

    print(f"Modifying the config.json file in {model_dir} ...")
    config_file = f"{model_dir}/config.json"
    embedding_model_name = "vllmLlamaEmbeddingModel"
    rename_config_model_arch(config_file, embedding_model_name)

    print("Registering the embedding model ...")
    always_true_detection = lambda architectures: True
    ModelRegistry.is_embedding_model = always_true_detection
    ModelRegistry.register_model(embedding_model_name, vllmLlamaEmbeddingModel)

    # TODO test out float32 here
    print("Wrapping vllm model with vllmLlamaWrapper ...")
    model = vllmLlamaWrapper(
        model_dir=model_dir,
        seed=args.seed,
        dtype=torch.float16)

    # Run test classification
    print(f"Running mteb tasks with {model_dir} ...")
    tasks = mteb.get_tasks(tasks=args.task_type)
    evaluation = mteb.MTEB(tasks=tasks)
    print("Running evaluation ...")
    results = evaluation.run(model,
                             verbosity=1,
                             overwrite_results=True,
                             output_folder=output_dir,
                             encode_kwargs={"batch_size": 32})

    for mteb_results in results:
        print(f"{args.model_type} on {mteb_results.task_name}: {mteb_results.get_score()}")

    print("Restoring the config.json file ...")
    rename_config_model_arch(config_file, "LlamaForCausalLM")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure the process group is destroyed upon exiting
        if dist.is_initialized():
            dist.destroy_process_group()