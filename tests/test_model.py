import unittest
import torch
import sys
import os

# Add project root to the Python path to allow importing from 'model'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.transformer import MultiHeadAttention, PositionalEncoding, EncoderLayer, GPT

class TestModelComponents(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.d_model = 64
        self.num_heads = 4
        self.d_ff = 128
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 100
        self.dropout = 0.1

        # Use a fixed seed for reproducibility
        torch.manual_seed(42)

    def test_multi_head_attention_shape(self):
        """Test if MultiHeadAttention's output has the correct shape."""
        attention = MultiHeadAttention(self.d_model, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = attention(x, x, x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_attention_weights_softmax(self):
        """Test if attention weights properly sum to 1."""
        attention = MultiHeadAttention(self.d_model, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # We can test the inner function directly
        Q = attention.W_q(x).view(self.batch_size, -1, attention.num_heads, attention.d_k).transpose(1, 2)
        K = attention.W_k(x).view(self.batch_size, -1, attention.num_heads, attention.d_k).transpose(1, 2)
        V = attention.W_v(x).view(self.batch_size, -1, attention.num_heads, attention.d_k).transpose(1, 2)
        
        _, attn_weights = attention.scaled_dot_product_attention(Q, K, V)
        
        # Check that weights sum to 1 across the key dimension for each query
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones(self.batch_size, self.num_heads, self.seq_len)))

    def test_positional_encoding_reproducibility(self):
        """Test that PositionalEncoding is stable and reproducible in eval mode."""
        pos_encoding = PositionalEncoding(self.d_model, max_seq_length=self.seq_len, dropout=0.1)
        pos_encoding.eval()  # Set to evaluation mode to disable dropout

        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        output1 = pos_encoding(x)
        output2 = pos_encoding(x)
        
        self.assertTrue(torch.equal(output1, output2), "PositionalEncoding should be deterministic in eval mode.")

    def test_positional_encoding_applies_dropout(self):
        """Test that PositionalEncoding applies dropout during training."""
        pos_encoding = PositionalEncoding(self.d_model, max_seq_length=self.seq_len, dropout=0.9)
        pos_encoding.train() # Set to training mode

        # With very high dropout, some outputs should be different from the non-dropout version
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        output_with_dropout = pos_encoding(x)
        
        # Manually calculate the output without dropout
        pe = pos_encoding.pe[:, :x.size(1), :]
        output_without_dropout = x + pe

        self.assertFalse(torch.equal(output_with_dropout, output_without_dropout), "Dropout should be applied in train mode.")

    def test_positional_encoding_shape(self):
        """Test the output shape of PositionalEncoding."""
        pos_encoding = PositionalEncoding(self.d_model, max_seq_length=self.seq_len)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = pos_encoding(x)
        self.assertEqual(output.shape, x.shape)

    def test_encoder_layer_shape(self):
        """Test the output shape of the EncoderLayer (Transformer Block)."""
        encoder_block = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = encoder_block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_encoder_layer_residuals_and_norm(self):
        """Check if layer norms and residuals are likely working."""
        encoder_block = EncoderLayer(self.d_model, self.num_heads, self.d_ff, dropout=0.0) # No dropout for this test
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = encoder_block(x)

        # The output of a LayerNorm should have mean close to 0 and std close to 1
        # This is an indirect way to check if the final norm was applied.
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)
        self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-6))
        self.assertTrue(torch.allclose(output_std, torch.ones_like(output_std), atol=1e-1))

    def test_gpt_model_forward_pass(self):
        """Test a full forward pass of the GPT model."""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=2, # Keep it small for a fast test
            d_ff=self.d_ff,
            max_seq_length=self.seq_len,
            dropout=self.dropout
        )
        # Input tensor with token IDs
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # The forward pass should execute without errors
        try:
            output = model(x)
            # Check the final output shape (logits)
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.vocab_size))
        except Exception as e:
            self.fail(f"GPT model forward pass failed with exception: {e}")

if __name__ == '__main__':
    unittest.main() 