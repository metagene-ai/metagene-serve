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
from typing import Optional, Set


class LlamaWrapper:
    def __init__(self,
                 model_dir,
                 model_dtype,
                 seed,
                 max_length=512):

        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = None
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        dtype = next(self.model.parameters()).dtype
        print(f"The model is loaded in: {dtype}")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name: str | None = None,  # Make task_name optional
               prompt_type: PromptType | None = None,
               **kwargs):

        set_seed(self.seed)
        batch_size = kwargs.get("batch_size", 32)

        embeddings = []

        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.model.device)

            # Remove `token_type_ids` if it exists
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

        return  np.array(embeddings)

class DNABERTWrapper:
    def __init__(self,
                 model_dir,
                 seed,
                 max_length=512):

        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = None
        self.seed = seed

        self.model = BertModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        dtype = next(self.model.parameters()).dtype
        print(f"The model is loaded in: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name: str | None = None,  # Make task_name optional
               prompt_type: PromptType | None = None,
               **kwargs):
        set_seed(self.seed)
        batch_size = kwargs.get("batch_size", 32)

        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

        return  np.array(embeddings)

class NTWrapper:
    def __init__(self,
                 model_dir,
                 seed,
                 max_length=512):

        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = None
        self.seed = seed

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        dtype = next(self.model.parameters()).dtype
        print(f"The model is loaded in: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name: str | None = None,  # Make task_name optional
               prompt_type: PromptType | None = None,
               **kwargs):
        set_seed(self.seed)
        batch_size = kwargs.get("batch_size", 32)

        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i: i + batch_size]

            tokens_ids = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length)["input_ids"]
            tokens_ids = tokens_ids.to(self.model.device)

            attention_mask = tokens_ids != self.tokenizer.pad_token_id
            with torch.no_grad():
                torch_outs = self.model(
                    tokens_ids,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                    output_hidden_states=True)

            batch_embeddings = torch_outs['hidden_states'][-1]
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            mean_sequence_embeddings = torch.sum(attention_mask * batch_embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
            embeddings.extend(mean_sequence_embeddings.cpu().to(torch.float32).numpy())

        return  np.array(embeddings)