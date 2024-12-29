import torch
from utils.data_processor import TextProcessor
from models.bigram import BigramLanguageModel

# hyperparameters
batch_size = 128
block_size = 256
max_iters = 20000
eval_interval = 2000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# initialize data processor
processor = TextProcessor('data/input.txt')
data = torch.tensor(processor.encode(processor.text), dtype=torch.long)

# train/val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


model = BigramLanguageModel(processor.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            train_x, train_y = [t.to(device) for t in get_batch('train')]
            val_x, val_y = [t.to(device) for t in get_batch('val')]
            train_loss = model(train_x, train_y)[1].item()
            val_loss = model(val_x, val_y)[1].item()
        print(f"step {iter}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}")
        model.train()

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(processor.decode(model.generate(
    context, max_new_tokens=500)[0].tolist()))
