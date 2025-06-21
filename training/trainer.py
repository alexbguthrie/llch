import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class LLMTrainer:
    def __init__(self, 
                 model, 
                 config=None,
                 train_dataset=None, 
                 val_dataset=None, 
                 batch_size=32,
                 learning_rate=3e-4,
                 max_epochs=10,
                 checkpoint_dir="checkpoints",
                 device=None):
        """
        Trainer for language models
        
        Args:
            model: The language model to train
            config: The configuration object (for saving checkpoints)
            train_dataset: Training dataset (optional, for training)
            val_dataset: Validation dataset (optional, for training)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_epochs: Maximum number of epochs to train
            checkpoint_dir: Directory to save model checkpoints
            device: Device to train on (will use CUDA if available if None)
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders only if datasets are provided
        if self.train_dataset:
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
        else:
            self.train_loader = None

        if self.val_dataset:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
        else:
            self.val_loader = None
            
        # Initialize optimizer and loss function only if training
        if self.train_dataset:
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.optimizer = None
            self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Get batch data
            src = batch["input_ids"].to(self.device)
            tgt = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(src)
            
            # Calculate loss
            # Reshape outputs and targets for loss calculation
            # outputs shape: (batch_size, seq_len, vocab_size)
            # tgt shape: (batch_size, seq_len)
            outputs = outputs.view(-1, outputs.size(-1))
            tgt = tgt.view(-1)
            
            loss = self.criterion(outputs, tgt)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        
        return avg_loss, elapsed
    
    def validate(self):
        """Evaluate on validation set"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Get batch data
                src = batch["input_ids"].to(self.device)
                tgt = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(src)
                
                # Calculate loss
                outputs = outputs.view(-1, outputs.size(-1))
                tgt = tgt.view(-1)
                loss = self.criterion(outputs, tgt)
                
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Train the model for max_epochs"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            
            # Train for one epoch
            train_loss, train_time = self.train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"Train loss: {train_loss:.4f} | Time: {train_time:.2f}s")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt")
                    print(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == self.max_epochs:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                
        # Plot training history
        self.plot_training_history()
                
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)

        # Convert config namespace to dict for saving
        config_dict = self.config_to_dict(self.config) if self.config else None

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': config_dict,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        # The filename from args.resume might already contain the directory.
        # We assume the user provides the full path if they are resuming.
        path = filename
        
        if not os.path.exists(path):
            # Fallback to checking inside the default checkpoint directory
            path_in_dir = os.path.join(self.checkpoint_dir, filename)
            if not os.path.exists(path_in_dir):
                print(f"Checkpoint not found at '{filename}' or in '{self.checkpoint_dir}'")
                raise FileNotFoundError(f"Checkpoint not found: {filename}")
            path = path_in_dir

        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Only load optimizer state if it exists (i.e., during training)
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Checkpoint loaded from {path}")
        
    def config_to_dict(self, namespace_obj):
        """Recursively convert a SimpleNamespace object to a dictionary."""
        if hasattr(namespace_obj, '__dict__'):
            return {k: self.config_to_dict(v) for k, v in namespace_obj.__dict__.items()}
        else:
            return namespace_obj

    def plot_training_history(self):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot to checkpoint directory
        if self.checkpoint_dir:
            save_path = os.path.join(self.checkpoint_dir, "training_history.png")
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    def generate_text(self, tokenizer, prompt, max_length=100, temperature=1.0):
        """Generate text from a prompt"""
        self.model.eval()
        
        input_ids = tokenizer.encode(prompt)
        
        # Generate tokens
        generated_ids = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Use the model's max_seq_length from its config
                max_seq_len = self.model.positional_encoding.pe.size(1)
                current_input = torch.tensor([generated_ids[-max_seq_len:]], dtype=torch.long, device=self.device)
                
                # Get model output
                outputs = self.model(current_input)
                
                # Get logits for the last token
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply softmax to get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample the next token
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS token is generated
                if next_token_id == tokenizer.special_tokens.get("<EOS>"):
                    break
                    
                generated_ids.append(next_token_id)
                
        # Decode the generated IDs
        return tokenizer.decode(generated_ids) 