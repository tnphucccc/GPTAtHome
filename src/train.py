import torch
from utils.data_processor import TextProcessor
from models.bigram import BigramLanguageModel
from models.gpt import GPTLanguageModel

# Hyperparameters
batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Context length for predictions
max_iters = 5000  # Total training iterations
eval_interval = 500  # How often to evaluate the model
learning_rate = 3e-4  # Optimizer learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Set random seed for reproducibility
torch.manual_seed(1337)

# Initialize data processor
processor = TextProcessor('data/input.txt')
data = torch.tensor(processor.encode(processor.text), dtype=torch.long)

# Split data into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Function to get a batch of data


def get_batch(split):
    """Get a batch of data from the training or validation set.

    Args:
        split (str): 'train' or 'val' to select the dataset

    Returns:
        tuple: Input and target tensors of shape (batch_size, block_size)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# Initialize model and optimizer
model = GPTLanguageModel(processor.vocab_size).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
"""
The training loop consists of the following steps:
1. Get a batch of training data
2. Compute model predictions and loss
3. Compute gradients and update model parameters
4. Periodically evaluate the model on the validation set

The model is trained for a fixed number of iterations, and the loss is printed
every eval_interval iterations.
"""
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

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': processor.vocab_size,
    'stoi': processor.stoi,
    'itos': processor.itos
}, 'model_checkpoint.pth')
