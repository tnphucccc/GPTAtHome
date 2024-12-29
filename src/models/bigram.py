import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """A simple bigram language model that predicts the next token based on the current token.

    This model uses an embedding table that maps each token to a vector of size vocab_size.
    During training, it optimizes the embeddings to predict the next token in a sequence.

    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer that maps token IDs to vectors
        embedding_proj (nn.Linear): Projection layer
        layer_norm (nn.LayerNorm): Layer normalization
        dropout (nn.Dropout): Dropout layer
    """

    def __init__(self, vocab_size, embed_dim=384, dropout_rate=0.1):
        """Initialize the bigram language model.

        Args:
            vocab_size (int): Size of the vocabulary - number of unique tokens
            embed_dim (int): Dimension of embeddings
            dropout_rate (float): Dropout probability
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.embedding_proj = nn.Linear(embed_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, idx, targets=None):
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target tensor of next tokens, shape (batch_size, sequence_length)

        Returns:
            tuple: (logits, loss) if targets provided, else logits
        """

        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # [B, T, embed_dim]
        token_emb = self.layer_norm(token_emb)
        token_emb = self.dropout(token_emb)
        logits = self.embedding_proj(token_emb)  # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

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
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat(
                [idx, torch.multinomial(probs, num_samples=1)], dim=-1)
        return idx
