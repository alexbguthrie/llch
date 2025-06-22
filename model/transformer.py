import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention with residual connection and layer norm
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 d_ff=2048, 
                 max_seq_length=5000, 
                 dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
    def generate_mask(self, src, tgt):
        # Source mask (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target mask (padding and subsequent mask)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        subsequent_mask = torch.triu(torch.ones((1, seq_length, seq_length), device=tgt.device), diagonal=1) == 0
        tgt_mask = tgt_mask & subsequent_mask
        
        return src_mask, tgt_mask
        
    def encode(self, src, src_mask):
        # Embed and add positional encoding
        src_embedded = self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model))
        
        # Pass through encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        return enc_output
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # Embed and add positional encoding
        tgt_embedded = self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model))
        
        # Pass through decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        return dec_output
    
    def forward(self, src, tgt):
        # Generate masks
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encode
        enc_output = self.encode(src, src_mask)
        
        # Decode
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Final linear layer
        output = self.fc(dec_output)
        
        return output

class GPT(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=768, 
                 num_heads=12, 
                 num_layers=12,
                 d_ff=3072, 
                 max_seq_length=1024, 
                 dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer layers (using EncoderLayer for self-attention only)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final linear layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def generate_mask(self, x):
        # Create causal mask to prevent attending to future tokens
        seq_length = x.size(1)
        # The mask should be (seq_length, seq_length), and it will be broadcasted 
        # correctly by PyTorch in the attention layer.
        # A value of `True` indicates a position that should be masked.
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
        return mask
        
    def forward(self, x):
        # Generate causal mask
        # The mask needs to be of shape (seq_length, seq_length) for the self-attention mechanism.
        # It will be broadcasted to (batch_size, num_heads, seq_len, seq_len) in the attention layer.
        mask = self.generate_mask(x)
        
        # Embed and add positional encoding
        x = self.positional_encoding(self.token_embedding(x) * math.sqrt(self.d_model))
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Final linear layer
        output = self.fc(x)
        return output

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_length=100, temperature=1.0):
        """
        Generate text from a prompt using the GPT model.
        This function handles the autoregressive generation loop.
        """
        self.eval()
        
        input_ids = tokenizer.encode(prompt)
        generated_ids = input_ids
        
        for _ in range(max_length):
            # The input to the model should not exceed its max sequence length
            current_input_ids = generated_ids[-self.positional_encoding.pe.size(1):]
            input_tensor = torch.tensor([current_input_ids], dtype=torch.long, device=next(self.parameters()).device)
            
            # Get model logits
            logits = self(input_tensor)
            
            # Get logits for the very last token and apply temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if EOS token is generated
            # Note: We need a robust way to get the EOS token ID.
            # This might need to be passed in or handled more gracefully.
            if tokenizer.special_tokens and next_token_id == tokenizer.special_tokens.get("<EOS>"):
                break
                
            generated_ids.append(next_token_id)
            
        return tokenizer.decode(generated_ids) 