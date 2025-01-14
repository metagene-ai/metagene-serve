import numpy as np
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
import torch
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig


if __name__ == "__main__":
    model_name_or_path = "metagene-ai/METAGENE-1"
    seed = 42
    set_seed(seed)

    override_pooler_config = PoolerConfig(
        pooling_type="MEAN",
        normalize=True,
        softmax=False)
    model = LLM(model=model_name_or_path,
                tokenizer=model_name_or_path,
                tokenizer_mode="auto",
                task="embed",
                override_pooler_config=override_pooler_config,
                seed=seed,
                dtype=torch.bfloat16,
                enforce_eager=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    sentences = [
        "ACTG",
        "CCCTAGC"
    ]

    embeddings = []
    for i in range(0, len(sentences), 1):
        batch = sentences[i:i + 1]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Remove `token_type_ids` if it exists
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        outputs = model.embed(inputs)
        # vllm return multiple outputs for one input with multiple request ids
        embedding = outputs[0].outputs.embedding
        embeddings.append(embedding)

    embeddings = np.array(embeddings)