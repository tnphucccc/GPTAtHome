import torch


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @torch.no_grad()
    def estimate_loss(self, get_batch, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train_step(self, X, Y):
        self.optimizer.zero_grad()
        logits, loss = self.model(X, Y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
