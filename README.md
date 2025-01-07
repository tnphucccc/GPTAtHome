# GPTAtHome

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A character-level language model implementation using PyTorch. Generates Shakespeare-style text using a bigram model architecture.

## Requirements

- Python 3.12+
- PyTorch with CUDA support (optional)
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/tnphucccc/GPTAtHome.git
cd GPTAtHome

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

# Start generating
python src/generate.py
```

## Project Structure
```bash
GPTAtHome/
├── data/
│   └── input.txt         # Training data (Shakespeare text)
├── src/
│   ├── models/
│   │   ├── bigram.py     # BigramLanguageModel implementation
│   │   └── gpt.py        # GPTLanguageModel implementation           
│   ├── utils/
│   │   └── data_processor.py
│   ├── train.py          # Training script
│   └── generate.py       # Generate script  
└── tests/
    ├── test_bigram.py    # Unit tests
    └── test_gpt.py
```
