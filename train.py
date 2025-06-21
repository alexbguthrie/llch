import os
import argparse
import torch
from model.transformer import GPT
from model.tokenizer import SimpleTokenizer, BPETokenizer
from data.dataset import TextDataset, TextFileDataset, WikiTextDataset
from training.trainer import LLMTrainer
from utils.data_utils import download_wikitext, load_text_files

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model from scratch')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gpt', choices=['gpt'], 
                        help='Type of model to train')
    parser.add_argument('--vocab_size', type=int, default=10000, 
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, 
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048, 
                        help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=512, 
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    
    # Tokenizer parameters
    parser.add_argument('--tokenizer_type', type=str, default='simple', choices=['simple', 'bpe'],
                        help='Type of tokenizer to use')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json',
                        help='Path to save/load tokenizer')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'text_files'],
                        help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data')
    parser.add_argument('--text_files_dir', type=str, default=None,
                        help='Directory containing text files (if dataset is text_files)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of epochs to train')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Generation parameters
    parser.add_argument('--generate', action='store_true',
                        help='Generate text after training')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Prompt for text generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for text generation')
    
    return parser.parse_args()

def create_tokenizer(args, texts=None):
    """Create or load a tokenizer"""
    if args.tokenizer_type == 'simple':
        tokenizer_class = SimpleTokenizer
    else:  # bpe
        tokenizer_class = BPETokenizer
        
    # Check if tokenizer already exists
    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = tokenizer_class.load(args.tokenizer_path)
    else:
        print(f"Creating new {args.tokenizer_type} tokenizer")
        tokenizer = tokenizer_class(vocab_size=args.vocab_size)
        
        # Train tokenizer if texts are provided
        if texts:
            print("Training tokenizer...")
            tokenizer.train(texts)
            
            # Save tokenizer
            tokenizer_dir = os.path.dirname(args.tokenizer_path)
            if tokenizer_dir:
                os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.save(args.tokenizer_path)
            print(f"Tokenizer saved to {args.tokenizer_path}")
            
    return tokenizer

def load_dataset(args, tokenizer):
    """Load and prepare dataset"""
    if args.dataset == 'wikitext':
        # Download WikiText-2 dataset if needed
        data_dir = download_wikitext(os.path.join(args.data_dir, 'wikitext-2'))
        
        # Create datasets
        train_dataset = WikiTextDataset(
            data_dir=data_dir,
            split='train',
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        val_dataset = WikiTextDataset(
            data_dir=data_dir,
            split='valid',
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    elif args.dataset == 'text_files':
        if args.text_files_dir is None:
            raise ValueError("text_files_dir must be specified when using text_files dataset")
            
        # Load text files
        texts = load_text_files(args.text_files_dir)
        
        if not texts:
            raise ValueError(f"No text files found in {args.text_files_dir}")
            
        # Split into train and validation
        split_idx = int(len(texts) * 0.9)  # 90% train, 10% validation
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        val_dataset = TextDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def create_model(args, vocab_size):
    """Create a language model"""
    if args.model_type == 'gpt':
        model = GPT(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_length=args.max_seq_length,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    return model

def run_training(args):
    """Run the model training process."""
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load or prepare data for tokenizer training
    if args.dataset == 'wikitext':
        data_dir = download_wikitext(os.path.join(args.data_dir, 'wikitext-2'))
        with open(os.path.join(data_dir, 'wiki.train.tokens'), 'r', encoding='utf-8') as f:
            sample_text = f.read(1000000)  # Use 1MB for tokenizer training
        texts = [sample_text]
    elif args.dataset == 'text_files':
        if args.text_files_dir is None:
            raise ValueError("text_files_dir must be specified for text_files dataset")
        texts = load_text_files(args.text_files_dir)
        if not texts:
            raise ValueError(f"No text files found in {args.text_files_dir}")
    else:
        texts = []

    # Create or load tokenizer
    tokenizer = create_tokenizer(args, texts)
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")

    # Load dataset
    train_dataset, val_dataset = load_dataset(args, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create model
    model = create_model(args, len(tokenizer.token_to_id))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create trainer
    trainer = LLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )

    # Resume training if checkpoint is provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train model
    trainer.train()

def run_generation(args):
    """Run the text generation process."""
    if not args.resume:
        raise ValueError("A checkpoint must be provided with --resume for generation.")
    if not args.tokenizer_path or not os.path.exists(args.tokenizer_path):
        raise ValueError("A valid tokenizer path must be provided with --tokenizer_path for generation.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    if args.tokenizer_type == 'simple':
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    else:
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")

    # Create model with the same architecture as the checkpoint
    model = create_model(args, len(tokenizer.token_to_id))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create a trainer instance just for loading the model and generation
    trainer = LLMTrainer(model=model, device=device, checkpoint_dir=args.checkpoint_dir)
    
    # Load the checkpoint
    trainer.load_checkpoint(args.resume)
    
    # Generate text
    print("\nGenerating text...")
    print(f"Prompt: {args.prompt}")
    generated_text = trainer.generate_text(
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    print(f"Generated text: {generated_text}")

def main():
    args = parse_args()
    
    if args.generate:
        run_generation(args)
    else:
        run_training(args)

if __name__ == '__main__':
    main() 