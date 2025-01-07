# GPTAtHome

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A character-level language model implementation using PyTorch. Generates Shakespeare-style text using bigram and gpt model architecture.

## Features

- Character-level language modeling
- Two model architectures: Bigram and GPT
- Shakespeare text generation
- Interactive text generation interface
- CUDA support for GPU acceleration

## Requirements

- Python 3.12+
- PyTorch with CUDA support (optional)
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tnphucccc/GPTAtHome.git
cd GPTAtHome
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Training the Model
Train the language model on Shakespeare text:
```bash
python src/train.py
```
This will:
- Load the Shakespeare dataset from `input.txt`.
- Train using the specified model architecture.
- Save the trained model checkpoint.

## Generating Text
Generate Shakespeare-style text using the trained model:
```bash
python src/generate.py
```
You can give your input context or just enter for random context.
Type `/quit` to exit the generation program.

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
