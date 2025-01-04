import torch
from models.bigram import BigramLanguageModel
from utils.data_processor import TextProcessor


def load_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        model = BigramLanguageModel(checkpoint['vocab_size'])
        model.load_state_dict(checkpoint['model_state_dict'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def generate_text(model, checkpoint, input_text="", max_tokens=500):
    device = next(model.parameters()).device

    if input_text:
        stoi = checkpoint['stoi']
        input_tokens = [stoi.get(c, 0) for c in input_text]
        context = torch.tensor([input_tokens], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    output_tokens = model.generate(
        context, max_new_tokens=max_tokens)[0].tolist()

    itos = checkpoint['itos']
    return ''.join([itos[i] for i in output_tokens])


if __name__ == "__main__":
    try:
        model, checkpoint = load_model('model_checkpoint.pth')

        print("Enter '/quit' to exit the program")
        while True:
            user_input = input(
                "Enter context (or press Enter for random start): ")

            if user_input.lower() == '/quit':
                print("\nExiting program...")
                break

            generated_text = generate_text(model, checkpoint, user_input)
            print("\nGenerated text:")
            print(generated_text)
            print("\n" + "="*50 + "\n")
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
