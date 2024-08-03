import unittest
from transformer import Transformer, AttentionBlock
from utils import softmax
import numpy as np


""""
class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = Transformer()

    def test_method1(self):
        result = self.transformer.method1()
        self.assertEqual(result, expected_result)

    def test_method2(self):
        result = self.transformer.method2()
        self.assertTrue(result)

    def test_method3(self):
        with self.assertRaises(ExpectedException):
            self.transformer.method3()
"""

"""
class TestAttentionBlock(unittest.TestCase):

    def setUp(self):
        self.attention_block = AttentionBlock()
"""


class TestUtils(unittest.TestCase):
    
    def test_softmax(self):
        result = softmax(np.array([1, 2, 3]))
        np.testing.assert_almost_equal(result, np.array([0.09, 0.24, 0.66]), decimal=2)
        self.assertEqual(sum(result), 1)


if __name__ == '__main__':
    unittest.main()