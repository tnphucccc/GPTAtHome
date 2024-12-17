import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def generate(self, idx, max_new_tokens):
        raise NotImplementedError
