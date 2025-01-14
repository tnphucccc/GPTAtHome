from dataclasses import dataclass
from pathlib import Path
import torch
import logging
from typing import Optional
from utils.data_processor import TextProcessor
from models.gpt import GPTLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path('data/input.txt')
MODEL_PATH = Path('checkpoints/model.pth')

# Set random seed for reproducibility
torch.manual_seed(1337)


@dataclass
class TrainConfig:
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4


class Trainer:
    def __init__(self, config: TrainConfig, processor: Optional[TextProcessor] = None):
        self.config = config
        self.processor = processor
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.best_val_loss = float('inf')

    def setup_data(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load and process training data"""
        data = torch.tensor(self.processor.encode(
            self.processor.text), dtype=torch.long)
        n = int(0.9 * len(data))
        return data[:n], data[n:], self.processor.vocab_size

    def get_batch(self, split: str, train_data: torch.Tensor,
                  val_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - self.config.block_size,
                           (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train(self) -> None:
        """Main training loop"""
        # Setup
        train_data, val_data, vocab_size = self.setup_data()
        model = GPTLanguageModel(vocab_size).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate)
        logger.info(f'Model parameters: {sum(p.numel()
                    for p in model.parameters())/1e6:.2f}M')

        # Training loop
        for iter in range(self.config.max_iters):
            # Training step
            model.train()
            xb, yb = self.get_batch('train', train_data, val_data)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Evaluation
            if ((iter + 1) % self.config.eval_interval == 0 or iter == 0):
                val_loss = self.evaluate(model, train_data, val_data)
                logger.info(f'Step {iter}: train loss {
                            loss.item():.4f}, val loss {val_loss:.4f}')
                self.save_checkpoint(model, val_loss)

    def evaluate(self, model: GPTLanguageModel, train_data: torch.Tensor,
                 val_data: torch.Tensor) -> float:
        """Evaluate model on validation set"""
        model.eval()
        with torch.no_grad():
            eval_x, eval_y = self.get_batch('val', train_data, val_data)
            _, val_loss = model(eval_x, eval_y)
        return val_loss.item()

    def save_checkpoint(self, model: GPTLanguageModel, val_loss: float) -> None:
        """Save model if validation loss improves"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': self.processor.vocab_size,
                'stoi': self.processor.stoi,
                'itos': self.processor.itos
            }, MODEL_PATH)
            logger.info(f'Saved checkpoint to {MODEL_PATH}')


def main():
    config = TrainConfig()
    processor = TextProcessor(DATA_PATH)
    trainer = Trainer(config, processor=processor)
    trainer.train()


if __name__ == '__main__':
    main()
