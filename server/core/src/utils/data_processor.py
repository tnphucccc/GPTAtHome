# Using the following command to download the input.txt file:
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

class TextProcessor:
    """
    Class for encoding and decoding text data.
    """

    def __init__(self, file_path):
        """
        Initialize the TextProcessor.

        Args:
            file_path (str): Path to the text file to process
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        """
        Encode a string into a list of integers.

        Args:
            s (str): Input string to encode
        """
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """
        Decode a list of integers into a string.

        Args:
            l (list): Input list of integers to decode
        """
        return ''.join([self.itos[i] for i in l])
