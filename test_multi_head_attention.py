from multi_head_attention import MultiHeadAttention
import torch

import unittest

class TestMultiHeadAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.num_heads = 8
        self.seq_len = 64
        self.batch_size = 64

        self.model = MultiHeadAttention(self.d_model, self.num_heads)

    def test_dimensionality(self):
        input, _ = torch.randn((self.batch_size, self.seq_len, self.d_model))
        output = self.model(input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    # TODO: Check test cases here: https://gemini.google.com/app/10517d319ccee364

if __name__ == '__main__':
    unittest.main()