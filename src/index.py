# main.py
import torch
from models.bigram_model import BigramLanguageModel
from utils.data_utils import prepare_data, get_batch
from trainer.trainer import Trainer

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


def main():
    # Load and prepare data
    with open('../data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    train_data, val_data = prepare_data(text, stoi)

    # Initialize model
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, device)

    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = trainer.estimate_loss(
                lambda split: get_batch(split, block_size, batch_size, device,
                                        train_data=train_data, val_data=val_data),
                eval_iters
            )
            print(f"step {iter}: train loss {
                  losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', block_size, batch_size, device,
                           train_data=train_data, val_data=val_data)
        trainer.train_step(xb, yb)

    context = "The meaning of life is"
    idx = torch.tensor([stoi[s] for s in context],
                       dtype=torch.long).unsqueeze(0).to(device)
    idx = model.generate(idx, 100)
    print(''.join([itos[i] for i in idx.squeeze().cpu().numpy()]))


if __name__ == '__main__':
    main()
