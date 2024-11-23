import mii
import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType
from huggingface_hub import login
login(token="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")


class DeepspeedMIIModel:
    def __init__(self, model_name: str):
        # Load the model using mii.pipeline
        self.model = mii.pipeline(model_name)

    def encode(
            self,
            sentences: list[str],
            task_name: str = "default_task",  # Provide a default value for task_name
            prompt_type: PromptType | None = None,
            **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task (default is "default_task").
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences as a NumPy array.
        """
        # Ensure we're using mixed precision
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use mixed precision if on a GPU
        with torch.autocast('cuda', dtype=torch.float16):
            # Filter out invalid keys like 'batch_size' and 'prompt_name'
            valid_kwargs = {k: v for k, v in kwargs.items() if k not in ['batch_size', 'prompt_name']}

            # Optionally, process the sentences if a prompt_type is provided
            if prompt_type:
                sentences = [prompt_type.apply(sentence) for sentence in sentences]

            # Encode the sentences using the mii pipeline
            embeddings = self.model(sentences, **valid_kwargs)

        # Return the embeddings as a numpy array
        return np.array(embeddings)

if __name__ == "__main__":
    # model_path = "meta-llama/Llama-3.1-8B"
    model_path = "../model_ckpts/safetensors/step-00078000/"
    model = DeepspeedMIIModel(model_path)

    task_name = "Banking77Classification"
    evaluation = mteb.MTEB(tasks=[task_name])

    print("running evaluation ...")
    results = evaluation.run(model, output_folder=f"results/{model_path}")
    print("evaluation results:", results)
