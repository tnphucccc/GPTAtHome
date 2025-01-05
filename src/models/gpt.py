import torch
import torch.nn as nn
import torch.nn.functional as F

n_embedding = 384
n_head = 6
n_layer = 6
dropout = 0.2
block_size = 256


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        key = self.key(x)
        query = self.query(x)

        # Compute attention scores
        attn = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        value = self.value(x)
        out = attn @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.fc = nn.Linear(n_head * head_size, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedFoward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embedding, n_head):
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedFoward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(
            *[Block(n_embedding, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.fc = nn.Linear(n_embedding, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.fc(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)

        return idx
