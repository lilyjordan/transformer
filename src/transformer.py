import numpy as np
import math
from utils import softmax


class Transformer:
    def __init__(self, key_dimension=64, value_dimension=64, model_dimension=512,
                 scaling_factor=10000, num_heads=8, num_layers=6,
                 max_sequence_length=512):
        """
        The key_dimension and value_dimension defaults are both chosen
        to be model_dimension divided by value_dimension (64 * 8 = 512),
        but that choice is somewhat arbitrary, so it isn't hard-coded
        into the initialization logic.

        I can't tell from the paper what they use for the max sequence length,
        or even whether they use a single uniform max sequence length; TODO
        figure that out?
        """
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.model_dimension = model_dimension
        self.scaling_factor = scaling_factor
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length


    def train(self):
        raise NotImplementedError


    def computePositionalEmbeddingMatrix(self):
        return np.fromfunction(
            np.vectorize(self.computePositionalEmbedding),
            (self.max_sequence_length, self.model_dimension)
        )


    def computePositionalEmbedding(self, token_index, embedding_index):
        if embedding_index % 2 == 0:
            return math.sin(self.scalePositionalEmbedding(token_index, embedding_index))
        else:
            return math.cos(self.scalePositionalEmbedding(token_index, embedding_index - 1))


    def scalePositionalEmbedding(self, token_index, embedding_index):
        return token_index / (
            self.scaling_factor ** (embedding_index / self.model_dimension)
        )


    def mask(self):
        raise NotImplementedError


class TransformerLayer:
    def __init__(self):
        pass


class AttentionBlock:
    def __init__(self):
        pass


    def computeAttention(self, queries, keys, values):
        dot_products = np.dot(queries, np.transpose(keys))
        key_dimension = keys.shape[1]
        # Scale down to mitigate vanishing gradients (speculatively; there may be some
        # debate over whether this would even happen?)
        scaled_dot_products = dot_products / math.sqrt(key_dimension)

        attention = softmax(scaled_dot_products) * values
        return attention


class FeedForwardNetwork:
    def __init__(self):
        pass