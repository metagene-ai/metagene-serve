import argparse
import mteb
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, AutoModelForMaskedLM
from transformers.trainer_utils import set_seed
import torch


class LlamaWrapper:
    def __init__(self,
                 model_name_or_path,
                 seed,
                 max_length=512):

        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name,
               prompt_type,
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
                return_tensors="pt").to(self.model.device)

            # Remove `token_type_ids` if it exists
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

        return np.array(embeddings)


class DNABERTWrapper:
    def __init__(self,
                 model_name_or_path,
                 seed,
                 max_length=512):

        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = BertModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name,
               prompt_type,
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
                return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

        return np.array(embeddings)


class NTWrapper:
    def __init__(self,
                 model_name_or_path,
                 seed,
                 max_length=512):

        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    # https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    def encode(self,
               sentences,
               task_name,
               prompt_type,
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

        return np.array(embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", nargs='+', required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    if isinstance(args.task_name, str):
        args.task_name = [args.task_name]
    args.model_name_or_path = args.model_name_or_path or "metagene-ai/METAGENE-1"
    args.seed = args.seed or 42

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Wrapping model with mtebWrapper ...")
    if args.model_name_or_path in [
        "zhihan1996/DNABERT-2-117M",
        "zhihan1996/DNABERT-S"
    ]:
        model = DNABERTWrapper(model_dir, args.seed)
    elif args.model_name_or_path in [
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "InstaDeepAI/NT-2.5b-1000g",
        "InstaDeepAI/NT-500m-1000g",
        "InstaDeepAI/NT-500m-human-ref",
        "InstaDeepAI/NT-v2-100m-multi-species",
        "InstaDeepAI/NT-v2-250m-multi-species",
        "InstaDeepAI/NT-v2-500m-multi-species",
        "InstaDeepAI/NT-v2-50m-multi-species"
    ]:
        model = NTWrapper(model_dir, args.seed)
    elif args.model_name_or_path == "metagene-ai/METAGENE-1":
        model = LlamaWrapper(model_dir, args.seed)
    else:
        raise ValueError(f"Invalid model name: {args.model_name_or_path}")

    print(f"Running mteb tasks with {args.model_type} ...")
    tasks = mteb.get_tasks(tasks=args.task_name)
    evaluation = mteb.MTEB(tasks=tasks)

    print("Running evaluation ...")
    results = evaluation.run(model, encode_kwargs={"batch_size": 32})
    for mteb_results in results:
        print(f"{args.model_type} on {mteb_results.task_name}: {mteb_results.get_score()}")


if __name__ == "__main__":
    main()