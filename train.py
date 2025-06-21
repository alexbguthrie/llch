import os
import argparse
import yaml
import torch
from types import SimpleNamespace

from model.transformer import GPT
from model.tokenizer import SimpleTokenizer, BPETokenizer
from data.dataset import TextDataset, TextFileDataset, WikiTextDataset
from training.trainer import LLMTrainer
from utils.data_utils import download_wikitext, load_text_files

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file and merge with command-line arguments."""
    # Load base config from YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Allow overriding with command-line arguments
    parser = argparse.ArgumentParser(description='Train or generate with a language model.')
    parser.add_argument('--config', type=str, default=config_path, help='Path to config file.')
    
    # Dynamically add arguments from the config file
    for section, params in config_dict.items():
        for param, value in params.items():
            # Use a different dest to avoid conflicts with nested structure
            cli_name = f'--{section}.{param}'
            if isinstance(value, bool) and not value:
                 parser.add_argument(cli_name, action='store_true', help=f'Override {param} in {section}')
            else:
                parser.add_argument(cli_name, type=type(value) if value is not None else str, default=value, help=f'Override {param} in {section}')

    args = parser.parse_args()

    # Merge YAML config with command-line overrides
    # This is a bit manual to handle the dot notation from argparse
    for section, params in config_dict.items():
        for param, default_value in params.items():
            cli_arg_name = f'{section}.{param}'
            if hasattr(args, cli_arg_name):
                cli_value = getattr(args, cli_arg_name)
                # Only update if the CLI value is different from the default,
                # to respect the hierarchy (CLI > YAML)
                if cli_value != default_value:
                    config_dict[section][param] = cli_value
    
    # Convert nested dictionaries to SimpleNamespace for dot notation access (e.g., config.model.d_model)
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)

    return dict_to_namespace(config_dict)

def create_tokenizer(config, texts=None):
    """Create or load a tokenizer based on the config."""
    if config.tokenizer.tokenizer_type == 'simple':
        tokenizer_class = SimpleTokenizer
    else:  # bpe
        tokenizer_class = BPETokenizer
        
    if os.path.exists(config.tokenizer.tokenizer_path):
        print(f"Loading tokenizer from {config.tokenizer.tokenizer_path}")
        tokenizer = tokenizer_class.load(config.tokenizer.tokenizer_path)
    else:
        print(f"Creating new {config.tokenizer.tokenizer_type} tokenizer")
        tokenizer = tokenizer_class(vocab_size=config.model.vocab_size)
        
        if texts:
            print("Training tokenizer...")
            tokenizer.train(texts)
            
            tokenizer_dir = os.path.dirname(config.tokenizer.tokenizer_path)
            if tokenizer_dir:
                os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.save(config.tokenizer.tokenizer_path)
            print(f"Tokenizer saved to {config.tokenizer.tokenizer_path}")
            
    return tokenizer

def load_data(config, tokenizer):
    """Load and prepare dataset based on the config."""
    if config.data.dataset == 'wikitext':
        data_dir = download_wikitext(os.path.join(config.data.data_dir, 'wikitext-2'))
        
        train_dataset = WikiTextDataset(
            data_dir=data_dir,
            split='train',
            tokenizer=tokenizer,
            max_length=config.model.max_seq_length
        )
        
        val_dataset = WikiTextDataset(
            data_dir=data_dir,
            split='valid',
            tokenizer=tokenizer,
            max_length=config.model.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    elif config.data.dataset == 'text_files':
        if config.data.text_files_dir is None:
            raise ValueError("text_files_dir must be specified when using text_files dataset")
            
        texts = load_text_files(config.data.text_files_dir)
        
        if not texts:
            raise ValueError(f"No text files found in {config.data.text_files_dir}")
            
        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        train_dataset = TextDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_length
        )
        
        val_dataset = TextDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

def create_model(config, vocab_size):
    """Create a language model based on the config."""
    if config.model.model_type == 'gpt':
        model = GPT(
            vocab_size=vocab_size,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.d_ff,
            max_seq_length=config.model.max_seq_length,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
        
    return model

def run_training(config):
    """Run the model training process."""
    if config.training.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)
    print(f"Using device: {device}")

    if config.data.dataset == 'wikitext':
        data_dir = download_wikitext(os.path.join(config.data.data_dir, 'wikitext-2'))
        with open(os.path.join(data_dir, 'wiki.train.tokens'), 'r', encoding='utf-8') as f:
            sample_text = f.read(1000000)
        texts = [sample_text]
    elif config.data.dataset == 'text_files':
        if config.data.text_files_dir is None:
            raise ValueError("text_files_dir must be specified for text_files dataset")
        texts = load_text_files(config.data.text_files_dir)
        if not texts:
            raise ValueError(f"No text files found in {config.data.text_files_dir}")
    else:
        texts = []

    tokenizer = create_tokenizer(config, texts)
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")

    train_dataset, val_dataset = load_data(config, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    model = create_model(config, len(tokenizer.token_to_id))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = LLMTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_epochs=config.training.max_epochs,
        checkpoint_dir=config.training.checkpoint_dir,
        device=device
    )

    if config.training.resume:
        trainer.load_checkpoint(config.training.resume)

    trainer.train()

def run_generation(config):
    """Run the text generation process."""
    # The primary config is now loaded from the checkpoint.
    # The config from the file is only used for a few settings.
    
    # --- 1. Load configuration from checkpoint ---
    checkpoint_path = config.training.resume
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError("A valid checkpoint path must be provided via training.resume")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain a 'config' key. Please re-train and save a new checkpoint.")

    # Convert config dict from checkpoint back to namespace
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    
    config_from_ckpt = dict_to_namespace(checkpoint['config'])

    # --- 2. Setup model and tokenizer based on saved config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer from {config_from_ckpt.tokenizer.tokenizer_path}")
    if config_from_ckpt.tokenizer.tokenizer_type == 'simple':
        tokenizer = SimpleTokenizer.load(config_from_ckpt.tokenizer.tokenizer_path)
    else:
        tokenizer = BPETokenizer.load(config_from_ckpt.tokenizer.tokenizer_path)
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")

    # Create model with the exact architecture from the checkpoint
    model = create_model(config_from_ckpt, len(tokenizer.token_to_id))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 3. Load model weights and generate ---
    trainer = LLMTrainer(model=model, device=device, checkpoint_dir=config_from_ckpt.training.checkpoint_dir)
    trainer.load_checkpoint(checkpoint_path) # Loads the weights
    
    # Use generation parameters from the provided config file, allowing overrides
    print("\nGenerating text...")
    print(f"Prompt: {config.generation.prompt}")
    generated_text = trainer.generate_text(
        tokenizer=tokenizer,
        prompt=config.generation.prompt,
        max_length=config.generation.max_length,
        temperature=config.generation.temperature
    )
    print(f"Generated text: {generated_text}")

def main():
    config = load_config()
    
    if config.generation.generate:
        run_generation(config)
    else:
        run_training(config)

if __name__ == '__main__':
    main() 