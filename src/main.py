import torch
from utils.data_processor import TextProcessor
from models.bigram import BigramLanguageModel

# hyperparameters
batch_size = 256
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 3e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        losses = []
        for _ in range(eval_interval):
            xb, yb = get_batch('val')
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        print(f"step {iter}: val loss {sum(losses)/len(losses):.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
# The model is trained to predict the next token given the current token
# So we start with a seed token and predict the next token
# Then we append the predicted token to the context and predict the next token
# We repeat this process to generate a sequence of tokens
for i in range(1000):
    logits, _ = model(context)
    probs = torch.softmax(logits[0, -1], dim=-1)
    context = torch.cat([context, torch.multinomial(
        probs, num_samples=1).unsqueeze(-1)], dim=-1)

print(processor.decode(context.squeeze().tolist()))
