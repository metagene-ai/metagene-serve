import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.distributed as dist
from tqdm import tqdm
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig


def cosine_similarity(A, B):
    A_flat = A.ravel()
    B_flat = B.ravel()

    dot_product = np.dot(A_flat, B_flat)

    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)

    if norm_A == 0.0 or norm_B == 0.0:
        return 0.0

    return dot_product / (norm_A * norm_B)


# Sample sentences
sentences = [
    "A",
    "CCGC",
    "TTGCA",
    "TTGCATTA",
    "GTATTACCGC",
    "TGCCTCCCGTACGT",
    "TAATGTTATATGCCCAGACGGACCCAGACCCGGGTCAGTCCAACTATAAGTACCCCTCGATGCTCCCCAACCCCCCGAATTTTTGTATGCGCGATTTATCATGTCTTCCACCATCCACCCTGTTGCCTGCAACCGCACGTGTGCCGATTACGTCAATCACGCGCGTCACGAGTGCTGCCTAGTTTGCCACCCTCCCTCCCCCACCCCAACTGTTGAGCATTGCCCAACATGTCGGTTGCCCGTCGATCTGTCGGAAGAACTCGCAAAGCTTTCCGCCGCGGTGGTCGAGCTGTCCGATCTCGTCGCCGATCTACACGTTTCTCTCGCCGGCGAAGAACTCGAGGACGCCGAGTCATGAGTACTCGGCGTATTCTCAATGTGACCTCTCGCAAGAAGGTCGACAATATGATGCCCATCGTCGTTGATGAGGAGTCTATTGTTACTGTGGGCCCCTTCACTTCGCCTTCCCCTCTCCTGTGTGTTTTCGTCCCCAACGCTAGGGATACCCGCACCCCCATCACAAACCCTGCCGTTCGGAATTCCTCGGACATCTTTGCAGTCGGTTACCGCGAAAAGGTCCGACTTGATGTCTTAGGTGGTGGAACCTTTATGTGGCGCAGGATTGTTTTCATGCTTAAGGGCGACGATCTTCGGCGCTTCATGGATTCCAGTAATTCTGGCAATATTCCTGCCCAGTTGTTCGATCAGACCACCGAGGGTGGCTGTCGACGTGTCATCGGCCCGCTTTTGGGCGTCACTAACGCCCAGACGGAGCTTCAAAAATATGTCTTCCGTGGTCAGGAAGACGTTGATTGGGCGGATCAATTTACGGCCCCCATCGACACTCGTCGTGTGACCGTTAAGTCCGACAAGATGCGGGTTATTCGGCCCGGTAATGAAACAGGGGCTTCCCGCCTATACCGTTTTTGGTATCCTATCCGCCGCACAATCTCTTACGAGGATGACCTCGAGAGTGATGTCGTCGGTGATCGGCCGTTCTCTACTGCCGGTTTACGGGGGGTGGGAGATATGTACGTCATGGATATTATGGGTATTACGAATTTAACCCCGGATGCACCGCAAACGTCCTACAGGTTCAGCCCTGAGGGCAGTTTTTACTGGCATGAGCGGTAAATTAGGAAACTATGGGGCTATCTAGGTACACGAAGACGCAGTTGGCATTAAGCCAATCTGTGTCAACACCAAGCTCGTCGCGTGGGTCGGAGTTAGATAGCCATATTGAGGGCCGAGCCCAGTGTACCAACTTTTTCCCCTTGTACTTGTCCGTGACATAGAACTGTTTCTGGTGACCTAACCAAAACTTGTAGCTCGGAAGGAACTTTATTCCTCCGAAGTCGTCAAAGATGGCGTATTCAATCCCATCAAGGTCCTCGTCCAGACTAAAGAGGCCTCCGAAGTAAGCATGCTTTCCCAAGCTTCGCGCCCAAACGGTTTTTCCCATTCGGGAAGGCCCGTATACCACGAGTGACTTTCTTCGTTCTGTATATCGAAGTTAGCGGTAATGACACGTAAGGGTCATTGACCCATCAGGGTGGGGTCCCCTATGGGGTGCCCTGTGGGGGCTGTACCCGCGGCAGACACCCCACTATAGGCCCGAGCGTAGCGAAGTTCACTTACCTCCAGTTTGACTTCCGACAAGATTTTCTCGTACCCACTCATCGAGTTCAGCCACCCAAGACGTGTCGATGTTAACTCCCTCCGGAGTCTCATACGGTACTCTGATGGGGGGGAATTTCCAGGCTGCATAAGCTCTGAGTTGGGTGAATGAAGTGACCAATGAACGTGGAGCCAGTGATTGGCATAGCGCCCAAAACTCTGACTCATTCTTTGCATTGATGATTTCAACCCACACCCCACCATTTGCATCCACTCTGCCTCCAGTAGGTCGTTCGAGTCCCCCAGCAACAACGTCTCCATCTTTGATTGCATAATCAAACCCGTCTTCTGGTGTACCACGTGACGGCGATACATTCGGGTGGCATCCTTCAACATCGAATGCACGGGCGTTCCTGGTCCGATATTTGACTCCGAAGTCGACAAAAGCGTGCAAATGAATACCCCCATCTGCGTGATCCTCTCGGCCGATGATGCATTCAGCTCCAAGTCCCGCAAGATGGTCGACCACTGCAAAAGGATCGAGGTCTCCACACTGAGGATAGGTAAGCAAGGCGTACCGTGCTTGAAATCGAAAAGTAGACATAGTTGGTTGTTGCACATTTTGCTTGGGTCCAAGTCTGGGCTTT",
]

def main():
    model_dir = "/expanse/lustre/projects/mia346/swang31/projects/MGFM/MGFM-serving/model_ckpts/safetensors/step-00086000"

    override_pooler_config = PoolerConfig(
        pooling_type="MEAN",
        normalize=False, # should be False for cosine similarity
        softmax=False) # should be False for cosine similarity
    llm = LLM(
        model=model_dir,
        task="embed",
        dtype=torch.float16,
        override_pooler_config=override_pooler_config) # non-override_pooler_config results in low cosine similarity

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    embeddings_vllm = []
    # have to process one sentence at a time, otherwise the vllm will internally batch the sentences and mess up the shape
    for i in range(0, len(sentences), 1):
        batch = sentences[i:i + 1]
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt")
        outputs = llm.embed(inputs)
        # vllm return multiple outputs for one input with multiple request ids
        embedding = outputs[0].outputs.embedding
        embeddings_vllm.append(embedding)

    del llm

    # baseline
    embeddings_baseline = []
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "auto")
    inputs = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings_baseline.extend(batch_embeddings.cpu().to(torch.float32).numpy())

    similarity = cosine_similarity(
        np.array(embeddings_vllm),
        np.array(embeddings_baseline))
    print("cosine similarity between embedding generation", similarity)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure the process group is destroyed upon exiting
        if dist.is_initialized():
            dist.destroy_process_group()