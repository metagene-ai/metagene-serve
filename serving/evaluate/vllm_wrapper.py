import os
import sys
sys.path.append("/home1/shangsha/workspace/MGFM/MGFM-serving/serving/evaluate/mteb")
import mteb
from mteb.encoder_interface import PromptType

import numpy as np
from sentence_transformers import SentenceTransformerModelCardData
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, AutoModelForMaskedLM
from transformers.trainer_utils import set_seed
import torch
from torch import nn
from typing import Iterable, List, Optional, Set, Tuple, Union
from vllm import LLM
from vllm import ModelRegistry
from vllm.attention import AttentionMetadata

from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.compressed_tensors.utils import get_compressed_tensors_cache_scale
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name

from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix


class vllmLlamaWrapper:
    def __init__(self,
                 model_dir,
                 seed=42,
                 dtype=torch.float16):
        self.model = LLM(model=model_dir,
                         tokenizer=model_dir,
                         tokenizer_mode="auto",
                         seed=seed,
                         dtype=dtype,
                         enforce_eager=True)
        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = None
        self.seed = seed

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
        set_seed(self.seed)
        outputs = self.model.encode(sentences)
        embeddings = [output.outputs.embedding for output in outputs]
        return  np.array(embeddings)

# https://github.com/vllm-project/vllm/blob/db100c5cdebc7140b57cbb40b20b5a28d7bff386/vllm/model_executor/models/llama.py#L631
class vllmLlamaEmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    """
    A model that uses Llama with additional embed functionalities.

    This class encapsulates the LlamaModel and provides an interface for
    embed operations and customized pooling functions.

    Attributes:
        model: An instance of LlamaModel used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
    }
    embedding_padding_modules = []

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()

        pooler_config = vllm_config.model_config.pooler_config

        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=True,
            softmax=False)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors, inputs_embeds)

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
                # Loading kv cache scales for compressed-tensors quantizate
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

                if "lm_head" in name:
                    continue
                if "input_layernorm" in name:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # LRUCacheWorkerLoRAManager instantiation requires model config.
    @property
    def config(self):
        return self.model.config