import os
import sys
sys.path.append("/home1/shangsha/workspace/MGFM/MGFM-serving/serving/evaluate/mteb")
import mteb
from mteb.encoder_interface import PromptType

import numpy as np
from sentence_transformers import SentenceTransformerModelCardData
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers.trainer_utils import set_seed
import torch
from torch import nn
from typing import Iterable, List, Optional, Set, Tuple, Union
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig


class vllmLlamaWrapper:
    def __init__(self,
                 model_dir,
                 seed=42,
                 dtype=torch.bfloat16):
        override_pooler_config = PoolerConfig(
            pooling_type="MEAN",
            normalize=True,
            softmax=False)
        self.model = LLM(model=model_dir,
                         tokenizer=model_dir,
                         tokenizer_mode="auto",
                         task="embed",
                         override_pooler_config=override_pooler_config,
                         seed=seed,
                         dtype=dtype,
                         enforce_eager=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
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

        set_seed(self.seed)

        embeddings = []
        for i in range(0, len(sentences), 1):
            batch = sentences[i:i + 1]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            # Remove `token_type_ids` if it exists
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            outputs = self.model.embed(inputs)
            # vllm return multiple outputs for one input with multiple request ids
            embedding = outputs[0].outputs.embedding
            embeddings.append(embedding)

        return np.array(embeddings)