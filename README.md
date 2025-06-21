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

Configuration for training and generation is handled by `config.yaml`. You can edit this file directly to change model parameters, paths, and hyperparameters.

**Option A: Train on WikiText-2 (Recommended for first use)**

The WikiText-2 dataset will be automatically downloaded. The default `config.yaml` is already set up for this. Simply run:

```bash
python3 train.py
```

**Option B: Train on Your Own Text Data**

1.  Edit `config.yaml` and change the following fields:
    - `data.dataset`: set to `'text_files'`
    - `data.text_files_dir`: set to the path of your data directory (e.g., `'my_data/'`)
    - `tokenizer.tokenizer_path`: choose a new path (e.g., `'tokenizers/bpe_custom.json'`)
    - `training.checkpoint_dir`: choose a new path (e.g., `'checkpoints/custom_model'`)

2.  Run the training script:
    ```bash
    python3 train.py
    ```

### 4. Generating Text

To generate text, you only need to provide a checkpoint path. The script automatically loads the model architecture and tokenizer path from the information saved in the checkpoint.

1.  Edit `config.yaml`:
    - Set `generation.generate` to `true`.
    - Set `training.resume` to the path of your trained model checkpoint (e.g., `'checkpoints/wikitext_model/best_model.pt'`).
    - You can also change the `generation.prompt`, `generation.max_length`, and `generation.temperature` to guide the output.

2.  Run the script:
    ```bash
    python3 train.py
    ```

### Overriding Config on the Command Line

For quick experiments, you can override any setting from `config.yaml` on the command line using dot notation:

```bash
# Run training with a different learning rate and batch size
python3 train.py --training.learning_rate 0.0001 --training.batch_size 16

# Run generation with a different prompt and temperature
python3 train.py --generation.generate --generation.prompt "Hello world" --generation.temperature 0.8
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
├── config.yaml           # Configuration file for training and generation.
├── README.md             # This file.
├── requirements.txt      # Project dependencies.
└── train.py              # The main script for training and generation.
```

## Command-Line Arguments

The `train.py` script is configured via `config.yaml`. You can also override any parameter on the command line.

*(For a full list of overridable parameters, see `config.yaml`)*

#### Key Training Arguments (override with `--section.key value`)
- `--data.dataset`: The dataset to use (`wikitext` or `text_files`).
- `--data.text_files_dir`: The directory containing your custom text files.
- `--model.d_model`, `--model.num_heads`, `--model.num_layers`: Key parameters to define the model's size and capacity.
- `--training.batch_size`, `--training.learning_rate`, `--training.max_epochs`: Core training loop parameters.
- `--training.resume`: Path to a checkpoint to resume training from.

#### Key Generation Arguments (override with `--section.key value`)
- `--generation.generate`: Flag to switch from training to generation mode.
- `--training.resume`: **(Required for generation)** Path to the model checkpoint. The model architecture and tokenizer path will be loaded from this file.
- `--generation.prompt`: The initial text to seed the generation.
- `--generation.max_length`: The maximum number of tokens in the generated output.
- `--generation.temperature`: Controls the randomness of the output. Higher values (e.g., 1.0) are more creative; lower values (e.g., 0.7) are more deterministic.

## License

This project is open source and available under the MIT License. 