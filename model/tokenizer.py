import re
import os
import json
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
    def train(self, texts):
        """Train the tokenizer on a list of texts"""
        # Simple word-level tokenization
        all_words = []
        for text in texts:
            # Split on whitespace and punctuation
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            all_words.extend(words)
            
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Take the most common words (minus the special tokens we already have)
        vocab_size = min(self.vocab_size, len(word_counts) + len(self.special_tokens))
        most_common = word_counts.most_common(vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
        for word, _ in most_common:
            idx = len(self.token_to_id)
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word
            
    def encode(self, text):
        """Convert text to token IDs"""
        # Simple word-level tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Convert to IDs
        ids = []
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                ids.append(self.special_tokens["<UNK>"])
                
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        words = []
        for idx in ids:
            if idx in self.id_to_token:
                words.append(self.id_to_token[idx])
                
        # Simple space joining (not perfect for punctuation)
        return " ".join(words)
    
    def save(self, path):
        """Save tokenizer vocabulary to disk"""
        tokenizer_dir = os.path.dirname(path)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
                "special_tokens": self.special_tokens
            }, f)
            
    @classmethod
    def load(cls, path):
        """Load tokenizer from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.special_tokens = data["special_tokens"]
        
        # Reconstruct id_to_token
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer


class BPETokenizer:
    """A simple implementation of Byte-Pair Encoding tokenizer"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        # Initialize with byte tokens (0-255)
        for i in range(256):
            char = bytes([i]).decode('latin-1')
            idx = len(self.token_to_id)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            
    def get_stats(self, words):
        """Count frequency of adjacent pairs"""
        pairs = {}
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def merge_pair(self, pair, words):
        """Merge all occurrences of a pair in the vocabulary"""
        new_words = []
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words.append(new_word)
            
        return new_words
    
    def train(self, texts, num_merges=None):
        """Train BPE on texts"""
        if num_merges is None:
            # Default to vocab_size - initial vocab size
            num_merges = self.vocab_size - len(self.token_to_id)

        # Build a frequency-counted vocabulary from the texts
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_counts.update(words)

        # Split each word into characters and store with its frequency
        # e.g., {'h u g': 5, 'p u g': 2}
        corpus = {' '.join(word): count for word, count in word_counts.items()}

        # Perform BPE merges
        for i in range(num_merges):
            # Get pair statistics from the current corpus
            pairs = Counter()
            for word, count in corpus.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += count
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create the new merged token
            merged_token = ''.join(best_pair)
            
            # Merge the pair in the corpus vocabulary
            new_corpus = {}
            bigram = ' '.join(best_pair)
            for word, count in corpus.items():
                new_word = word.replace(bigram, merged_token)
                new_corpus[new_word] = count
            corpus = new_corpus

            # Add the merged token to the vocabulary
            if merged_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_token] = idx
                self.id_to_token[idx] = merged_token
                
            # Record the merge
            self.merges[best_pair] = merged_token
            
            # Stop if we've reached the target vocabulary size
            if len(self.token_to_id) >= self.vocab_size:
                break
                
    def encode(self, text):
        """Encode text using learned BPE merges"""
        # A simple pre-tokenization scheme: split by spaces and punctuation
        # This preserves spaces as tokens
        pre_tokenizer_regex = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\S+"""
        words = re.findall(pre_tokenizer_regex, text)

        # Start with characters, handling spaces correctly
        word_tokens = [' '.join(list(word)) for word in words]
        
        # Apply merges
        for word_idx, word in enumerate(word_tokens):
            tokens = word.split()
            
            # Apply merges until no more can be applied
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
                merged = False
                
                for pair in pairs:
                    if pair in self.merges:
                        idx = pairs.index(pair)
                        tokens = tokens[:idx] + [self.merges[pair]] + tokens[idx+2:]
                        merged = True
                        break
                        
                if not merged:
                    break
                    
            word_tokens[word_idx] = ' '.join(tokens)
            
        # Convert to token IDs
        ids = []
        for word in word_tokens:
            for token in word.split():
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    ids.append(self.special_tokens["<UNK>"])
                    
        return ids
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
                
        # Join tokens (this is a simplification)
        text = "".join(tokens)
        
        # Replace the BPE-specific space representation with a standard space
        text = text.replace(" ", " ")
        return text.replace("</w>", " ")

    def detokenize(self, text):
        """A simple de-tokenizer to clean up the generated text."""
        # Remove space before punctuation
        text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
        return text.strip()
    
    def save(self, path):
        """Save tokenizer to disk"""
        tokenizer_dir = os.path.dirname(path)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Convert tuple keys in merges to strings for JSON serialization
        merges_str_keys = {str(k): v for k, v in self.merges.items()}
        
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
                "merges": merges_str_keys,
                "special_tokens": self.special_tokens
            }, f)
            
    @classmethod
    def load(cls, path):
        """Load tokenizer from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        
        # Convert stringified tuple keys back to tuples
        merges_str_keys = data.get("merges", {})
        tokenizer.merges = {tuple(eval(k)): v for k, v in merges_str_keys.items()}
        
        tokenizer.special_tokens = data["special_tokens"]
        
        # Reconstruct id_to_token
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer 