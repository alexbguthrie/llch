import os
import torch
from torch.utils.data import Dataset
import random

class TextDataset(Dataset):
    def __init__(self, 
                 texts, 
                 tokenizer, 
                 max_length=512,
                 stride=256):
        """
        Dataset for language modeling
        
        Args:
            texts: List of text documents
            tokenizer: Tokenizer to encode the texts
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Process all texts into examples
        self.examples = []
        self._process_texts()
        
    def _process_texts(self):
        """Process all texts into overlapping chunks"""
        for text in self.texts:
            # Encode the text
            token_ids = self.tokenizer.encode(text)
            
            # Create overlapping chunks
            for i in range(0, len(token_ids) - 1, self.stride):
                # Extract chunk of appropriate length
                chunk = token_ids[i:i + self.max_length]
                
                # Skip short chunks at the end
                if len(chunk) < self.max_length // 2:
                    continue
                    
                # Pad if necessary
                if len(chunk) < self.max_length:
                    chunk = chunk + [self.tokenizer.token_to_id["<PAD>"]] * (self.max_length - len(chunk))
                    
                # Create example
                self.examples.append({
                    "input_ids": chunk[:-1],  # Input is all tokens except last
                    "labels": chunk[1:]       # Target is all tokens except first (shifted by 1)
                })
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }

class TextFileDataset(TextDataset):
    def __init__(self, 
                 file_path, 
                 tokenizer, 
                 max_length=512,
                 stride=256):
        """
        Dataset for language modeling from a text file
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to encode the texts
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
        """
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Initialize with the text
        super().__init__([text], tokenizer, max_length, stride)

class WikiTextDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 split='train', 
                 tokenizer=None,
                 max_length=512):
        """
        Dataset for WikiText data
        
        Args:
            data_dir: Directory containing WikiText data
            split: 'train', 'valid', or 'test'
            tokenizer: Tokenizer to encode the texts
            max_length: Maximum sequence length
        """
        assert split in ['train', 'valid', 'test'], "Split must be 'train', 'valid', or 'test'"
        
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.token_to_id.get("<PAD>", 0)
        
        # Load data
        file_path = os.path.join(data_dir, f"wiki.{split}.tokens")
        self.examples = self._load_and_process_wiki_data(file_path)
        
    def _process_article(self, article_lines, examples):
        """Tokenizes, chunks, and pads a single article, adding it to examples."""
        article_text = ' '.join(article_lines)
        if not article_text:
            return

        tokens = self.tokenizer.encode(article_text)
        
        # Create chunks of max_length
        for i in range(0, len(tokens), self.max_length - 1):
            chunk = tokens[i:i + self.max_length]
            
            # Skip short chunks at the end
            if len(chunk) < self.max_length // 2:
                continue
                
            # Pad the last chunk if it's shorter than max_length
            if len(chunk) < self.max_length:
                padded = chunk + [self.pad_token_id] * (self.max_length - len(chunk))
            else:
                padded = chunk

            examples.append({
                "input_ids": padded[:-1],
                "labels": padded[1:]
            })

    def _load_and_process_wiki_data(self, file_path):
        """Load and process WikiText data by splitting it into articles."""
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_article = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # A new article starts with a heading like '= Article Title ='
            is_heading = line.startswith('= ') and line.endswith(' =')
            
            if is_heading and current_article:
                self._process_article(current_article, examples)
                current_article = [] # Reset for the new article
            
            current_article.append(line)
                    
        # Process the last article in the file
        if current_article:
            self._process_article(current_article, examples)
                        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        } 