# LLM From Scratch

This project is a from-scratch implementation of a Transformer-based language model, similar in style to GPT. It is written in Python using PyTorch and provides a complete toolkit for training and running your own language models. The project is designed to be clear, modular, and extensible.

## Features

- **Transformer Model**: A GPT-style decoder-only transformer architecture.
- **Customizable Tokenizers**: Includes a simple word-level tokenizer and a more advanced Byte-Pair Encoding (BPE) tokenizer.
- **Flexible Data Loading**: Supports automatic downloading of the WikiText-2 dataset or training on your own custom text files.
- **Comprehensive Training**: A dedicated `LLMTrainer` class handles the entire training loop, including validation, checkpointing, and logging.
- **Text Generation**: Generate new text from your trained models with adjustable creativity (temperature).
- **Extensible Design**: Easily add new model architectures, datasets, or tokenizers.

## Getting Started

Follow these steps to set up the project, train a model, and generate text.

### 1. Prerequisites

- Python 3.8 or higher
- Git for cloning the repository

### 2. Setup

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/alexbguthrie/llch.git
cd llch
```

Next, it is highly recommended to create and activate a Python virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (on macOS/Linux)
source .venv/bin/activate

# On Windows, use:
# .venv\Scripts\activate
```

Now, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Training a Model

You can train a model on the standard WikiText-2 dataset or on your own text files.

**Option A: Train on WikiText-2 (Recommended for first use)**

The WikiText-2 dataset will be automatically downloaded. This command trains a BPE tokenizer and then starts training the model.

```bash
# Train a new model on the WikiText-2 dataset
# --tokenizer_type bpe: Uses the advanced BPE tokenizer
# --tokenizer_path tokenizers/bpe_wikitext.json: Saves the trained tokenizer
# --checkpoint_dir checkpoints/wikitext_model: Saves model checkpoints here
python train.py \
    --dataset wikitext \
    --tokenizer_type bpe \
    --tokenizer_path tokenizers/bpe_wikitext.json \
    --checkpoint_dir checkpoints/wikitext_model
```

For a smaller, faster-training model, you can reduce the model dimensions:

```bash
# Train a smaller model for quicker testing
python train.py \
    --dataset wikitext \
    --tokenizer_type bpe \
    --tokenizer_path tokenizers/bpe_wikitext_small.json \
    --checkpoint_dir checkpoints/wikitext_model_small \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 4 \
    --max_seq_length 256
```

**Option B: Train on Your Own Text Data**

Place your text files (e.g., `.txt` files) in a directory. The script will load all files from that directory.

```bash
# Train a model on custom text files located in the 'my_data/' directory
python train.py \
    --dataset text_files \
    --text_files_dir my_data/ \
    --tokenizer_type bpe \
    --tokenizer_path tokenizers/bpe_custom.json \
    --checkpoint_dir checkpoints/custom_model
```

### 4. Generating Text

Once you have a trained model, you can use it to generate text. You must provide the path to the tokenizer and the model checkpoint.

```bash
# Generate text using a trained model
# --generate: Activates generation mode
# --resume: Path to the saved model checkpoint (use 'best_model.pt' for the best one)
# --prompt: The starting text for the model
# --max_length: The total number of tokens in the generated output
python train.py \
    --generate \
    --tokenizer_path tokenizers/bpe_wikitext.json \
    --resume checkpoints/wikitext_model/best_model.pt \
    --prompt "The future of AI is" \
    --max_length 100
```

## Project Structure

```
llm_from_scratch/
├── data/
│   └── dataset.py        # Dataset classes for handling text data.
├── model/
│   ├── tokenizer.py      # Implementations for Simple and BPE tokenizers.
│   └── transformer.py    # Core Transformer and GPT model architectures.
├── training/
│   └── trainer.py        # The LLMTrainer class orchestrating the training process.
├── utils/
│   └── data_utils.py     # Utilities for downloading and preprocessing data.
├── tests/
│   └── test_tokenizers.py # Unit tests for the tokenizers.
├── .gitignore            # Files and directories to be ignored by Git.
├── CHANGELOG.md          # A log of changes made to the project.
├── README.md             # This file.
├── requirements.txt      # Project dependencies.
└── train.py              # The main script for training and generation.
```

## Command-Line Arguments

The `train.py` script is highly configurable via command-line arguments.

*(For a full list of arguments, run `python train.py --help`)*

#### Key Training Arguments
- `--dataset`: The dataset to use (`wikitext` or `text_files`).
- `--data_dir`: The base directory for storing data.
- `--text_files_dir`: The directory containing your custom text files.
- `--model_type`: The model architecture to use (currently `gpt`).
- `--tokenizer_type`: The tokenizer to use (`simple` or `bpe`).
- `--tokenizer_path`: Where to save/load the trained tokenizer.
- `--checkpoint_dir`: Where to save model checkpoints.
- `--d_model`, `--num_heads`, `--num_layers`: Key parameters to define the model's size and capacity.
- `--batch_size`, `--learning_rate`, `--max_epochs`: Core training loop parameters.
- `--resume`: Path to a checkpoint to resume training from.

#### Key Generation Arguments
- `--generate`: Flag to switch from training to generation mode.
- `--prompt`: The initial text to seed the generation.
- `--max_length`: The maximum number of tokens in the generated output.
- `--temperature`: Controls the randomness of the output. Higher values (e.g., 1.0) are more creative; lower values (e.g., 0.7) are more deterministic.

## License

This project is open source and available under the MIT License. 