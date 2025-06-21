# LLM From Scratch

This project implements a Language Model from scratch using PyTorch. It includes a transformer-based architecture similar to GPT, along with tokenizers, data loaders, and training utilities.

## Project Structure

```
llm_from_scratch/
├── data/                 # Data handling code
│   └── dataset.py        # Dataset classes for text data
├── model/                # Model architecture
│   ├── tokenizer.py      # Tokenizer implementations
│   └── transformer.py    # Transformer model architecture
├── training/             # Training utilities
│   └── trainer.py        # Trainer class for model training
├── utils/                # Utility functions
│   └── data_utils.py     # Data downloading and preprocessing
├── requirements.txt      # Project dependencies
├── train.py              # Main training script
└── README.md             # This file
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Components

### Transformer Architecture

The implementation includes:
- **Multi-head self-attention**: Allows the model to focus on different parts of the input sequence
- **Positional encoding**: Provides information about token positions in the sequence
- **Feed-forward networks**: Processes representations from attention layers
- **Layer normalization**: Stabilizes training by normalizing activations
- **Residual connections**: Helps with gradient flow during training

### Tokenizers

Two tokenizer implementations are provided:
1. **SimpleTokenizer**: A word-level tokenizer for quick experiments
2. **BPETokenizer**: A Byte-Pair Encoding tokenizer for subword tokenization

### Datasets

The project supports:
- **WikiText-2 dataset**: Automatically downloaded and processed
- **Custom text files**: Train on your own text data

### Training

The `LLMTrainer` class handles:
- Model training and validation
- Checkpointing and model saving
- Loss tracking and visualization
- Text generation from trained models

## Usage

### Training a Model

To train a GPT-style language model on the WikiText-2 dataset:

```bash
python train.py --dataset wikitext --model_type gpt --tokenizer_type bpe
```

For a smaller model that trains faster:

```bash
python train.py --d_model 256 --num_heads 4 --num_layers 4 --max_seq_length 256
```

### Using Your Own Text Data

To train on your own text files:

```bash
python train.py --dataset text_files --text_files_dir path/to/your/text/files
```

### Generating Text

To generate text after training:

```bash
python train.py --generate --prompt "Once upon a time" --max_length 100 --temperature 0.8 --resume checkpoints/best_model.pt
```

## Command Line Arguments

### Model Parameters
- `--model_type`: Type of model to train (default: 'gpt')
- `--vocab_size`: Vocabulary size (default: 10000)
- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--dropout`: Dropout rate (default: 0.1)

### Tokenizer Parameters
- `--tokenizer_type`: Type of tokenizer to use ('simple' or 'bpe', default: 'simple')
- `--tokenizer_path`: Path to save/load tokenizer (default: 'tokenizer.json')

### Data Parameters
- `--dataset`: Dataset to use for training ('wikitext' or 'text_files', default: 'wikitext')
- `--data_dir`: Directory containing the data (default: 'data')
- `--text_files_dir`: Directory containing text files (if dataset is text_files)

### Training Parameters
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--max_epochs`: Maximum number of epochs to train (default: 10)
- `--checkpoint_dir`: Directory to save model checkpoints (default: 'checkpoints')
- `--device`: Device to train on ('cuda' or 'cpu')
- `--resume`: Path to checkpoint to resume training from

### Generation Parameters
- `--generate`: Generate text after training
- `--prompt`: Prompt for text generation (default: 'Once upon a time')
- `--max_length`: Maximum length of generated text (default: 100)
- `--temperature`: Temperature for text generation (default: 1.0)

## How the Model Works

This implementation builds a transformer-based language model similar to GPT:

1. **Tokenization**: Text is converted into tokens using either a simple word-level tokenizer or BPE
2. **Embedding**: Tokens are embedded into a continuous vector space
3. **Positional Encoding**: Position information is added to the embeddings
4. **Self-Attention**: The model learns which parts of the input to focus on
5. **Feed-Forward Networks**: Further process the attention outputs
6. **Output Layer**: Projects to vocabulary size for next-token prediction

The model is trained using a causal language modeling objective, where it learns to predict the next token given the previous tokens.

## Extending the Project

### Adding a New Model Architecture

1. Create a new model class in `model/transformer.py`
2. Add the model type to the choices in `train.py`
3. Update the `create_model` function in `train.py`

### Adding a New Dataset

1. Create a new dataset class in `data/dataset.py`
2. Add the dataset type to the choices in `train.py`
3. Update the `load_dataset` function in `train.py`

## License

This project is open source and available under the MIT License. 