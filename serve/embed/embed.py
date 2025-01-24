import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import set_seed
import torch


if __name__ == "__main__":
    set_seed(42)

    model_name_or_path = "metagene-ai/METAGENE-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "auto")

    model.eval()

    sentences = [
        "ACTG",
        "CCCTAGC"
    ]

    embeddings = []

    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(model.device)

        # Remove `token_type_ids` if it exists
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

    embeddings = np.array(embeddings)