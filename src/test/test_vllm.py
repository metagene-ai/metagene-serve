from vllm import (
    LLM,
    SamplingParams
)
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders
)


def rebuild_tokenizer(model_path):
    print("Create a BPE tokenizer ...")
    tokenizer = Tokenizer(models.BPE.from_file(
        f"{model_path}/vocab.json",
        f"{model_path}/merges.txt")
    )
    tokenizer.normalizer = normalizers.NFD()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
    tokenizer.save(f"{model_path}/fixed_tokenizer.json")

    print("Change to Transformers format ...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{model_path}/fixed_tokenizer.json"
    )
    fast_tokenizer.save_pretrained(model_path)

    print("Tokenizer rebuilt")

def test_backend_decoding():
    # Access the backend_tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model vocab size:", model.config.vocab_size)

    assert model.config.vocab_size == model.get_input_embeddings().weight.size(0), "Tokenizer vocab size != model vocab size"
    assert tokenizer.backend_tokenizer is not None, "Tokenizer backend is not initialized."

    tokens = tokenizer.tokenize("ACCGTTGCCGGCGTACTCCCCAGGTGGATAGCTTAATGGTTTCCCTCAGGCACCC")
    assert tokenizer.backend_tokenizer.decoder is not None, "Tokenizer backend decoder value is null."

    ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = tokenizer.decode(ids)
    print(f"Tokens: {tokens}, IDs: {ids}, Decoded: {decoded_text}")


if __name__ == "__main__":
    model_path = "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/step-00078000/"

    # generate results from vllm
    prompts = [
        "ATTTCACCGC",
        "TGCCTCCCGTAGG",
        "TCATTATGCAAAAGGC",
        "GTATTACCGCGGCTGCTGGC",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    print("Loading vllm test model ...")
    llm = LLM(model=model_path, tokenizer=model_path)
    print("Test model loaded")

    print("Generating sample outputs ...")
    outputs = llm.generate(prompts)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
