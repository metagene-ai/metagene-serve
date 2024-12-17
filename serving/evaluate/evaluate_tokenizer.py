from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing, Sequence
from transformers import AutoModel, PreTrainedTokenizerFast, AutoModelForCausalLM
import torch


def main():
    examples = [
        "TCATTATGCAAAAGGC",
        "_TCATTATGCAAAAGGC"
    ]

    model_path = "/project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/step-00078000"
    model = AutoModelForCausalLM.from_pretrained(
        model_path).to("cuda")
    first_param = next(model.parameters())
    print(f"The model is loaded in: {first_param.dtype} on {first_param.device}")
    model.eval()

    tokenizer_folder = "/home1/shangsha/workspace/MGFM/MGFM-serving/test/tokenizer"
    file_list = [
        "tokenizer_rebuilt.json",
        "tokenizer_rebuilt_prepend.json",
        "tokenizer_rebuilt_bos.json",
        "tokenizer_rebuilt_prepend_bos.json"

    ]

    for example in examples:
        print(f"\nExample: {example}")

        for file in file_list:
            tokenizer = Tokenizer.from_file(f"{tokenizer_folder}/{file}")
            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

            print(f"\nTokenizer: {file}")

            # Normalization
            if tokenizer.normalizer is not None:
                normalized = tokenizer.normalizer.normalize_str(example)
                print("1) Normalized:", normalized)
            else:
                normalized = example
                print("1) Normalized not defined:", normalized)

            # Pre-tokenization
            pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(normalized)
            print("2) Pre-tokenized:", pre_tokens)

            # Model encoding (no special tokens)
            raw_encoding = tokenizer.encode(example, add_special_tokens=False)
            print("3) Raw encoding:", raw_encoding.tokens)
            print("   Raw encoding IDs:", raw_encoding.ids)

            # Post-processing
            processed_encoding = tokenizer.post_processor.process(raw_encoding, None)
            print("4) Post-processed tokens:", processed_encoding.tokens)
            print("   Post-processed token IDs:", processed_encoding.ids)

            encoded = tokenizer.encode(example)
            print(f"5) Direct encoding: {encoded.tokens}")

            inputs = fast_tokenizer(
                example,
                return_tensors="pt",
                add_special_tokens=True,
                return_token_type_ids=False).to("cuda")
            print("6) Fast token IDs:", inputs["input_ids"])
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=42)
                decoded_text = fast_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("7) Model generation:", decoded_text)


if __name__ == "__main__":
    main()