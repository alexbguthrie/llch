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
    """A BPE tokenizer inspired by the GPT-2 implementation."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}  # (t1, t2) -> rank
        self.special_tokens = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        }
        self.end_of_word_token = "</w>"
        
        # Initialize with special tokens, basic bytes, and end-of-word token
        initial_vocab = list(self.special_tokens.keys())
        initial_vocab.extend(bytes([i]).decode('latin-1') for i in range(256))
        initial_vocab.append(self.end_of_word_token)
        
        for token in sorted(list(set(initial_vocab))):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
            
    def train(self, texts, num_merges=None):
        if num_merges is None:
            num_merges = self.vocab_size - len(self.token_to_id)

        # Use a regex that handles spaces and punctuation well
        pre_tokenizer_regex = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\w]+| ?[^\s\w]+"""
        
        # Get word counts
        word_counts = Counter()
        for text in texts:
            word_counts.update(re.findall(pre_tokenizer_regex, text))
        
        # Initialize the corpus with character-split words and their counts
        corpus = {' '.join(list(word)) + f' {self.end_of_word_token}': count for word, count in word_counts.items()}

        for i in range(num_merges):
            # Count pairs in the current corpus
            pairs = Counter()
            for word, count in corpus.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j+1])] += count
            
            if not pairs: break
            
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            merged_token = ''.join(best_pair)
            
            # Apply the merge to the corpus
            new_corpus = {word.replace(' '.join(best_pair), merged_token): count for word, count in corpus.items()}
            corpus = new_corpus

            # Add new token to vocabulary
            if merged_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_token] = idx
                self.id_to_token[idx] = merged_token
            
            # Record the merge operation with its rank (order of creation)
            self.merges[best_pair] = i
            
            if len(self.token_to_id) >= self.vocab_size: break
                
    def encode(self, text):
        pre_tokenizer_regex = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\w]+| ?[^\s\w]+"""
        words = re.findall(pre_tokenizer_regex, text)
        
        all_ids = []
        for word in words:
            # Start with characters and add the end-of-word token
            tokens = list(word) + [self.end_of_word_token]
            
            while len(tokens) > 1:
                # Find all current pairs and their ranks from the learned merges
                pairs = list(zip(tokens[:-1], tokens[1:]))
                merge_ranks = {pair: self.merges.get(pair, float('inf')) for pair in pairs}
                
                # Find the rank of the next best merge
                best_rank = min(merge_ranks.values())
                if best_rank == float('inf'):
                    break # No more possible merges for this word
                
                # Find the first pair that has the best rank
                best_pair = min(merge_ranks, key=merge_ranks.get)
                
                # Merge the best pair
                merged_token = ''.join(best_pair)
                new_tokens = []
                i = 0
                while i < len(tokens):
                    # If we find the best pair, merge it and skip the next token
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # Convert final subword tokens to IDs
            all_ids.extend(self.token_to_id.get(token, self.special_tokens["<UNK>"]) for token in tokens)
            
        return all_ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(idx, "") for idx in ids]
        text = "".join(tokens)
        # Replace the end-of-word token with a space and clean up
        text = text.replace(self.end_of_word_token, " ").strip()
        return text
    
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
                "special_tokens": self.special_tokens,
                "end_of_word_token": self.end_of_word_token,
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
        tokenizer.end_of_word_token = data.get("end_of_word_token", "</w>")
        
        # Reconstruct id_to_token
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer 