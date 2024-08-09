import numpy as np
import math
from utils import softmax, relu, xavier_initialize


class Transformer:
    def __init__(self, key_dimension=64, value_dimension=64, model_dimension=512,
                 feed_forward_dimension=2048, scaling_factor=10000, num_heads=8,
                 num_layers=6, max_sequence_length=512):
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
        self.feed_forward_dimension = feed_forward_dimension
        self.scaling_factor = scaling_factor
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

    def train(self):
        raise NotImplementedError

    def forwardPass(self):
        raise NotImplementedError

    def backprop(self):
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


class TransformerLayer:
    def __init__(self, num_heads, model_dimension):
        self.multi_head_attention = MultiHeadAttention(num_heads, model_dimension)
        self.feed_forward_network = FeedForwardNetwork()
    # three components:
    # multihead attention
    # add and norm
    # feedforward
    """
    "We employ a residual connection [11] around each of
    the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
    LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
    itself."
    """
    def forwardPass(self):
        pass


    @staticmethod
    def computeResidual(input, sublayer_function):
        return input + sublayer_function(input)

    @staticmethod
    def normalize(weights):
        EPSILON = 1e-8
        sqrt_variance = np.sqrt(np.var(weights))
        return weights - np.mean(weights) / sqrt_variance + EPSILON


class MultiHeadAttention:
    def __init__(self, num_heads, model_dimension):
        self.num_heads = num_heads
        self.model_dimension = model_dimension
        self.heads = []
        for _ in range(self.num_heads):
            self.heads.append(AttentionHead())

    def computeMultiHeadAttention(
            self,
            queries,
            keys,
            values,
            query_projection,
            key_projection,
            value_projection,
            overall_projection
        ):
        concatenated_attentions = np.array([])
        for head in self.heads:
            single_head_attention = head.computeAttention(
                np.dot(queries, query_projection),
                np.dot(keys, key_projection),
                np.dot(values, value_projection)
            )
            concatenated_attentions.append(single_head_attention)
        return np.dot(concatenated_attentions, overall_projection)


class AttentionHead:
    def __init__(self):
        pass

    def computeAttention(self, queries, keys, values, mask=True):
        dot_products = np.dot(queries, np.transpose(keys))
        key_dimension = keys.shape[1]
        # Scale down to mitigate vanishing gradients (speculatively; there may be some
        # debate over whether this would even happen?)
        scaled_dot_products = dot_products / math.sqrt(key_dimension)
        if mask:
            scaled_dot_products = self.causal_mask(scaled_dot_products)

        softmaxes = np.apply_along_axis(softmax, 1, scaled_dot_products)
        attention = np.dot(softmaxes, values)
        return attention

    def causal_mask(self, weights):
        """
        "We implement this inside of scaled dot-product attention by masking out
        (setting to −∞) all values in the input of the softmax which correspond to
        illegal connections."
        intuitively, you shouldn't be able to query a key where the key is at a later
        position than the query
        """
        causal_mask = np.fromfunction((lambda x, y: x < y), weights.shape)
        masked_weights = np.ma.masked_array(weights, mask=causal_mask)
        filled = masked_weights.filled(-np.inf)
        return filled


class FeedForwardNetwork:
    def __init__(self, model_dimension, feed_forward_dimension):
        self.model_dimension = model_dimension

        self.hidden_layer_weights = xavier_initialize(model_dimension, feed_forward_dimension)
        self.hidden_layer_biases = np.zeros(feed_forward_dimension)

        self.output_layer_weights = xavier_initialize(feed_forward_dimension, model_dimension)
        self.output_layer_biases = np.zeros(model_dimension)

    def forwardPass(self, input):
        if len(input) != self.model_dimension:
            raise ValueError(f"""Input array has length {len(input)} but should have
                            dimensions {self.model_dimension}""")

        hidden_layer_pre_activation = (np.dot(self.hidden_layer_weights, input) +
                                       self.hidden_layer_biases)
        hidden_layer_post_activation = np.vectorize(relu)(hidden_layer_pre_activation)
        output = (np.dot(self.output_layer_weights, hidden_layer_post_activation) +
                  self.output_layer_biases)
        return output