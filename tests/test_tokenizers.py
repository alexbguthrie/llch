import unittest
import os
import sys
import shutil

# Add project root to the Python path to allow importing from 'model'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.tokenizer import SimpleTokenizer, BPETokenizer

class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        """Set up a tokenizer and a directory for test files."""
        self.texts = ["hello world", "hello there"]
        self.tokenizer = SimpleTokenizer(vocab_size=20)
        self.test_dir = "test_tokenizer_files"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_train_and_encode(self):
        """Test that the tokenizer trains and encodes correctly."""
        self.tokenizer.train(self.texts)
        self.assertIn("hello", self.tokenizer.token_to_id)
        self.assertIn("world", self.tokenizer.token_to_id)
        self.assertIn("there", self.tokenizer.token_to_id)
        
        encoded = self.tokenizer.encode("hello world")
        self.assertEqual(len(encoded), 2)
        
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded.replace(" ", ""), "helloworld")

    def test_save_and_load(self):
        """Test that the tokenizer can be saved and loaded."""
        self.tokenizer.train(self.texts)
        save_path = os.path.join(self.test_dir, "simple_tokenizer.json")
        self.tokenizer.save(save_path)
        
        self.assertTrue(os.path.exists(save_path))
        
        loaded_tokenizer = SimpleTokenizer.load(save_path)
        self.assertEqual(self.tokenizer.token_to_id, loaded_tokenizer.token_to_id)
        
        encoded_original = self.tokenizer.encode("hello there")
        encoded_loaded = loaded_tokenizer.encode("hello there")
        self.assertEqual(encoded_original, encoded_loaded)

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        """Set up a BPE tokenizer and a test directory."""
        self.texts = ["hug", "pug", "pun", "bun", "hugs"]
        # Vocab size needs to be larger than initial tokens (256 bytes + 4 special) to allow merges
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.test_dir = "test_bpe_tokenizer_files"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_train_and_encode(self):
        """Test that the BPE tokenizer trains and encodes correctly."""
        self.tokenizer.train(self.texts)
        # Check if a common pair like ('u', 'g') has been merged
        self.assertIn(('u', 'g'), self.tokenizer.merges)
        
        encoded = self.tokenizer.encode("hugs")
        # Should be encoded as ['h', 'ugs'] or similar after merges
        self.assertTrue(len(encoded) < 4)
        
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("hugs", decoded.replace(" ", ""))

    def test_save_and_load(self):
        """Test that the BPE tokenizer can be saved and loaded."""
        self.tokenizer.train(self.texts)
        save_path = os.path.join(self.test_dir, "bpe_tokenizer.json")
        self.tokenizer.save(save_path)
        
        self.assertTrue(os.path.exists(save_path))
        
        loaded_tokenizer = BPETokenizer.load(save_path)
        self.assertEqual(self.tokenizer.merges, loaded_tokenizer.merges)
        
        encoded_original = self.tokenizer.encode("pun")
        encoded_loaded = loaded_tokenizer.encode("pun")
        self.assertEqual(encoded_original, encoded_loaded)

if __name__ == '__main__':
    unittest.main() 