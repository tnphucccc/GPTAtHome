import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):
    """A simple bigram language model that predicts the next token based on the current token.

    This model uses an embedding table that maps each token to a vector of size vocab_size.
    During training, it optimizes the embeddings to predict the next token in a sequence.

    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer that maps token IDs to vectors
    """

    def __init__(self, vocab_size):
        """Initialize the bigram language model.

        Args:
            vocab_size (int): Size of the vocabulary - number of unique tokens
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target tensor of next tokens, shape (batch_size, sequence_length)

        Returns:
            tuple:
                - logits (torch.Tensor): Raw model outputs, shape (batch_size, sequence_length, vocab_size)
                - loss (torch.Tensor or None): Cross entropy loss if targets provided, else None
        """

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens by sampling from the model's probability distribution.

        Takes an input context of tokens and autoregressively generates max_new_tokens
        additional tokens by sampling from the predicted probability distribution.

        Args:
            idx (torch.Tensor): Starting token indices, shape (batch_size, seq_length)
            max_new_tokens (int): Number of new tokens to generate

        Returns:
            torch.Tensor: Concatenated sequence of input and generated tokens,
                        shape (batch_size, seq_length + max_new_tokens)

        Generation process:
            1. Get logits from model for current sequence
            2. Take last token's predictions
            3. Convert to probabilities with softmax
            4. Sample next token from probability distribution
            5. Append new token to sequence
            6. Repeat until max_new_tokens reached
        """

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx = torch.cat(
                [idx, torch.multinomial(probs, num_samples=1)], dim=-1)
        return idx
