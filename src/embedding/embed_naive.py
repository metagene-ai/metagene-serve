import argparse
import mteb
from mteb.encoder_interface import PromptType
import numpy as np
from sentence_transformers import SentenceTransformerModelCardData
import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, ModelCard
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.compressed_tensors.utils import get_compressed_tensors_cache_scale
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput


class vllmLlamaEmbeddingModel(nn.Module):
    """Llama's vllm wrapper with additional embedding functionalities.

   Attributes:
       model: An instance of LlamaModel used for forward => generates token-level embeddings.
       _pooler: An instance of Pooler used for pooling =>  combines token-level embeddings.
   """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = LlamaModel(**kwargs)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    # https://github.com/vllm-project/vllm/blob/db100c5cdebc7140b57cbb40b20b5a28d7bff386/vllm/model_executor/models/llama.py#L318
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None, # add None default value
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model.forward(input_ids, positions, kv_caches, attn_metadata, inputs_embeds)

    # https://github.com/vllm-project/vllm/blob/db100c5cdebc7140b57cbb40b20b5a28d7bff386/vllm/model_executor/models/llama.py#L683
    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    # https://github.com/vllm-project/vllm/blob/db100c5cdebc7140b57cbb40b20b5a28d7bff386/vllm/model_executor/models/llama.py#L353
    def load_weights(
            self,
            weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                # added line to skip additional lm_head
                # KeyError: 'lm_head.weight'
                if "lm_head" in name:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class mtebWrapper:
    def __init__(self, model):
        self.model = model
        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = None

    def encode(
            self,
            sentences: list[str],
            task_name: str | None = None,  # Make task_name optional
            prompt_type: PromptType | None = None,
            **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        outputs = self.model.encode(sentences)
        embeddings = [output.outputs.embedding for output in outputs]
        return embeddings

def rename_config_model_arch(config_file, name):
    with open(config_file, 'r+') as file:
        data = json.load(file)
        data["architectures"] = name
        file.seek(0)
        json.dump(data, file, indent=2)
        file.truncate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    model_path = args.model_path or "/project/MGFM/MGFM-serving/model_ckpts/safetensors/step-00078000"

    # remember to modify the config.json file
    config_file = f"{model_path}/config.json"
    embedding_model_name = "vllmLlamaEmbeddingModel"
    rename_config_model_arch(config_file, embedding_model_name)

    # Regist our model to "fool" vllm
    always_true_detection = lambda architectures: True
    ModelRegistry.is_embedding_model = always_true_detection
    ModelRegistry.register_model(embedding_model_name, vllmLlamaEmbeddingModel)

    # Loading the model via vllm and apply the wrapper for mteb
    print("Loading model via vllm and wrapped with mtebWrapper ...")
    model = LLM(model=model_path, tokenizer=model_path, enforce_eager=True)
    model = mtebWrapper(model)

    # Run test classification
    print("Running mteb tasks ...")
    task_name = "Banking77Classification"
    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model)

    # modify the config.json file to its default value
    rename_config_model_arch(config_file, "LlamaForCasual")
