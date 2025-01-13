import os

import requests
import torch

from core.src.models.gpt import GPTLanguageModel


class RuntimeModel:
    def __init__(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weight/model.pth"
        )
        if os.path.exists(path) is False:
            RuntimeModel.download_model(path)

        self.model, self.checkpoint = self.load_model(path)

    def load_model(self, checkpoint_path: str):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path, weights_only=True)
            else:
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            

            model = GPTLanguageModel(checkpoint["vocab_size"])
            model.load_state_dict(checkpoint["model_state_dict"])

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            return model, checkpoint
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def download_model(path: str):
        url = (
            "https://github.com/tnphucccc/GPTAtHome/releases/download/v1.0.1/model.pth"
        )

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Model downloaded successfully to {path}")
        else:
            print(f"Failed to download model, status code: {response.status_code}")

    def generate_text(self, input_text: str = "", max_tokens: int = 2000):
        device = next(self.model.parameters()).device

        if input_text:
            stoi = self.checkpoint["stoi"]
            input_tokens = [stoi.get(c, 0) for c in input_text]
            context = torch.tensor([input_tokens], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        output_tokens = self.model.generate(context, max_new_tokens=max_tokens)[
            0
        ].tolist()

        itos = self.checkpoint["itos"]
        return "".join([itos[i] for i in output_tokens])

    def request(self, prompt: str, max_tokens: int = 2000):
        return self.generate_text(prompt, max_tokens)
