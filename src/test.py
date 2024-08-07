import unittest
from transformer import Transformer, AttentionHead
from utils import softmax
import numpy as np


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = Transformer(model_dimension=8)

    def testComputePositionalEmbeddingOdd(self):
        token_index = 30
        embedding_index = 1
        self.transformer.computePositionalEmbedding(token_index, embedding_index)
        # TODO check against reference values

    def testComputePositionalEmbeddingEven(self):
        token_index = 30
        embedding_index = 2
        self.transformer.computePositionalEmbedding(token_index, embedding_index)
        # TODO check against reference values

    def testComputePositionalEmbeddingMatrix(self):
        result = self.transformer.computePositionalEmbeddingMatrix()
        self.assertEqual(result.shape, (self.transformer.max_sequence_length, self.transformer.model_dimension))
        # TODO check against reference values

    def testScalePositionalEmbedding(self):
        """
          30 * 10000^(2/8)
        = 30 / 10000^(1/4)
        = 30 / 10
        = 3
        """
        token_index = 30
        embedding_index = 2
        result = self.transformer.scalePositionalEmbedding(token_index, embedding_index)
        self.assertEqual(result, 3)


class TestAttentionHead(unittest.TestCase):

    def setUp(self):
        self.attention_head = AttentionHead(model_dimension=8, scaling_factor=10000, max_sequence_length=10)

    def testComputeAttentionOutputShape(self):
        test_key_dimension = 4
        test_value_dimension = 5
        test_sequence_length = 6

        queries = np.random.randn(test_sequence_length, test_key_dimension)
        keys = np.random.randn(test_sequence_length, test_key_dimension)
        values = np.random.randn(test_sequence_length, test_value_dimension)

        result = self.attention_head.computeAttention(queries, keys, values)
        np.testing.assert_equal(result.shape, np.array([test_sequence_length, test_value_dimension]))

    def testComputeAttention(self):
        """
        Test against Pytorch as a reference implementation
        """
        """
        queries = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        keys = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 2, 2]])
        values = np.array([[2, 4, 6, 8], [12, 14, 16, 18], [22, 24, 26, 28]])
        """
        queries = np.array([
            [ 2.9412,  0.1560,  1.1144,  0.7364],
            [ 0.4442,  0.3593,  0.2383, -0.4030],
            [-0.5874,  0.8019, -0.6198,  0.6114],
            [ 1.0542, -1.0063, -2.1706, -0.2161],
            [ 0.8360, -0.0373, -1.3386,  0.1979],
            [ 0.5253, -1.1018, -0.1069,  0.1975]])

        keys = np.array([
            [ 1.7076, -0.3796, -0.1499, -0.8828],
            [ 0.8816, -0.6995, -1.3343, -0.7614],
            [-1.3195, -0.9910,  0.2598,  0.1719],
            [ 0.1234, -1.0525, -0.2646, -1.2752],
            [-0.1808,  0.0161,  0.1208, -0.5637],
            [-1.3620, -0.5759,  0.2375, -0.8744]])

        values = np.array([
            [-0.5517, -0.8557,  1.0039,  0.5527, -0.0837],
            [ 0.2323,  0.1714,  0.1891, -1.5772,  0.4250],
            [-2.1781, -0.1613, -1.1233, -0.3182,  1.8093],
            [ 0.8523,  3.6962,  1.4628,  1.3894,  1.6764],
            [-0.3334,  0.0062, -0.2643, -0.5679,  1.0708],
            [-0.7923, -1.7509, -0.6533,  0.8814, -1.0792]])

        expected = np.array([
            [-0.3964, -0.4284,  0.8071,  0.2725,  0.1639],
            [-0.3342,  0.1957,  0.2946,  0.0933,  0.5618],
            [-0.6414, -0.0336, -0.1483, -0.0657,  0.6587],
            [ 0.0326,  0.4023,  0.4525, -0.5602,  0.5330],
            [-0.1366,  0.2537,  0.3984, -0.3301,  0.5333],
            [-0.3391,  0.3840,  0.2874,  0.0289,  0.6813]])

        result = self.attention_head.computeAttention(queries, keys, values) 
        np.testing.assert_almost_equal(result, expected, decimal=2)
  

class TestUtils(unittest.TestCase):
    
    def testSoftmax(self):
        result = softmax(np.array([1, 2, 3]))
        np.testing.assert_almost_equal(result, np.array([0.09, 0.24, 0.66]), decimal=2)
        self.assertEqual(sum(result), 1)


if __name__ == '__main__':
    unittest.main()