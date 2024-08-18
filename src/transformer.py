import numpy as np
import math
from utils import softmax, relu, xavier_initialize
from transformers import GPT2TokenizerFast


class Transformer:
    def __init__(
        self,
        key_dimension=64,
        value_dimension=64,
        model_dimension=512,
        feed_forward_dimension=2048,
        scaling_factor=10000,
        num_heads=8,
        num_layers=6,
        max_sequence_length=200,
    ):
        """
        "Absolutely unmixed attention is prayer." -Simone Weil

        The key_dimension and value_dimension defaults are both chosen
        to be model_dimension divided by value_dimension (64 * 8 = 512),
        but that choice is somewhat arbitrary, so it isn't hard-coded
        into the initialization logic.

        I can't tell from the paper what they use for the max sequence length,
        or even whether they use a single uniform max sequence length; TODO
        figure that out?
        Looks like 512 is standard (I think? double check?), or that's what
        they used or something, but I'm setting to 200 for the time being
        for testing so as not to confuse it with the model dimension.
        """
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.model_dimension = model_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.scaling_factor = scaling_factor
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(
                TransformerLayer(
                    self.num_heads,
                    self.key_dimension,
                    self.value_dimension,
                    self.model_dimension,
                    self.feed_forward_dimension,
                    self.max_sequence_length,
                )
            )

        #  TODO are these dimensions right?
        self.embeddings = xavier_initialize(
            in_dimension=model_dimension, out_dimension=1
        )

        self.embedding_layer = EmbeddingLayer(self.model_dimension)

    def train(self, inputs):
        #  TODO create a batch_size instance variable in the constructor
        #  and then split the input into batches
        raise NotImplementedError

    def train_on_batch(self, batch):
        # TODO implement
        pass

    def forwardPass(self, input_embedding):
        output = input_embedding
        for layer in self.layers:
            output = layer.forwardPass(output)
        #  TODO there's something in here called "linear" in the architecture diagram
        return softmax(output)

    def computeGradient(self, point, function, epsilon=1e-8):
        # TODO define one consistent epsilon
        return function(point + epsilon) - function(point - epsilon) / (2 * epsilon)

    def updateWeights(self):
        """
        "If we turn our mind toward the good, it is impossible that little by little the
        whole soul will not be attracted thereto in spite of itself." -Simone Weil
        """
        #  TODO I think we'll want off-the-shelf Adam for this
        raise NotImplementedError

    def computePositionalEncodingMatrix(self):
        return np.fromfunction(
            np.vectorize(self.computePositionalEncoding),
            (self.max_sequence_length, self.model_dimension),
        )

    def computePositionalEncoding(self, token_index, embedding_index):
        if embedding_index % 2 == 0:
            return math.sin(self.scalePositionalEncoding(token_index, embedding_index))
        else:
            return math.cos(
                self.scalePositionalEncoding(token_index, embedding_index - 1)
            )

    def scalePositionalEncoding(self, token_index, embedding_index):
        return token_index / (
            self.scaling_factor ** (embedding_index / self.model_dimension)
        )


class EmbeddingLayer:
    def __init__(self, model_dimension):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.embeddings = xavier_initialize(self.tokenizer.vocab_size, model_dimension)

    def forwardPass(self, input):
        tokens = self.tokenize(input)
        return self.embeddings[tokens]

    def tokenize(self, input):
        """
        "A mind enclosed in language is in prison." -Simone Weil
        """
        tokens = self.tokenizer(input)["input_ids"]
        return tokens


class TransformerLayer:
    def __init__(
        self,
        num_heads,
        key_dimension,
        value_dimension,
        model_dimension,
        feed_forward_dimension,
        max_sequence_length,
    ):
        self.multi_head_attention = MultiHeadAttention(
            num_heads,
            model_dimension,
            key_dimension,
            value_dimension,
            max_sequence_length,
        )
        self.feed_forward_network = FeedForwardNetwork(
            model_dimension, feed_forward_dimension
        )

    def forwardPass(self, input):
        attention = self.multi_head_attention.computeMultiHeadAttention(input)
        added_and_normed_attention = self.add_and_norm(input, attention)

        feed_forward_output = self.feed_forward_network.forwardPass(
            added_and_normed_attention
        )
        added_and_normed_feed_forward_output = self.normalize(feed_forward_output)

        return added_and_normed_feed_forward_output

    @staticmethod
    def normalize(weights):
        EPSILON = 1e-8
        sqrt_variance = np.sqrt(np.var(weights))
        return weights - np.mean(weights) / sqrt_variance + EPSILON

    def add_and_norm(self, layer_input, layer_output):
        residual = layer_input + layer_output
        normed = self.normalize(residual)
        return normed


class MultiHeadAttention:
    def __init__(
        self,
        num_heads,
        model_dimension,
        key_dimension,
        value_dimension,
        max_sequence_length,
    ):
        self.num_heads = num_heads
        self.model_dimension = model_dimension

        self.query_weights = xavier_initialize(model_dimension, model_dimension)
        self.key_weights = xavier_initialize(model_dimension, model_dimension)
        self.value_weights = xavier_initialize(model_dimension, model_dimension)

        self.heads = []
        for _ in range(self.num_heads):
            self.heads.append(
                AttentionHead(
                    model_dimension,
                    key_dimension,
                    value_dimension,
                )
            )

        self.overall_projection = xavier_initialize(
            value_dimension * num_heads, model_dimension
        )

    def computeMultiHeadAttention(self, input):
        """
        "Attention alone, that attention which is so full that the 'I' disappears, is
        required of me." -Simone Weil
        """
        concatenated_attentions = np.array([])
        head_outputs = []
        for head in self.heads:
            single_head_attention = head.forwardPass(
                input,
                self.query_weights,
                self.key_weights,
                self.value_weights,
            )
            head_outputs.append(single_head_attention)
            concatenated_attentions = np.concatenate(head_outputs, axis=1)
        return np.dot(concatenated_attentions, self.overall_projection)


class AttentionHead:
    def __init__(
        self,
        model_dimension,
        key_dimension,
        value_dimension,
    ):
        self.query_projection = xavier_initialize(model_dimension, key_dimension)
        self.key_projection = xavier_initialize(model_dimension, key_dimension)
        self.value_projection = xavier_initialize(model_dimension, value_dimension)
        pass

    def forwardPass(self, input, query_weights, key_weights, value_weights):
        queries = np.dot(input, query_weights)
        keys = np.dot(input, key_weights)
        values = np.dot(input, value_weights)

        projected_queries = np.dot(queries, self.query_projection)
        projected_keys = np.dot(keys, self.key_projection)
        projected_values = np.dot(values, self.value_projection)

        return self.computeAttention(
            projected_queries, projected_keys, projected_values
        )

    def computeAttention(self, queries, keys, values, mask=True):
        dot_products = np.dot(queries, np.transpose(keys))
        key_dimension = keys.shape[1]
        # Scale down to mitigate vanishing gradients (speculatively; there may be some
        # debate over whether this would even happen?)
        scaled_dot_products = dot_products / math.sqrt(key_dimension)
        if mask:
            scaled_dot_products = self.causalMask(scaled_dot_products)

        softmaxes = np.apply_along_axis(softmax, 1, scaled_dot_products)
        attention = np.dot(softmaxes, values)
        return attention

    def causalMask(self, weights):
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

        self.hidden_layer_weights = xavier_initialize(
            model_dimension, feed_forward_dimension
        )
        self.hidden_layer_biases = np.zeros(feed_forward_dimension)

        self.output_layer_weights = xavier_initialize(
            feed_forward_dimension, model_dimension
        )
        self.output_layer_biases = np.zeros(model_dimension)

    def forwardPass(self, input):
        hidden_layer_pre_activation = (
            np.dot(input, self.hidden_layer_weights) + self.hidden_layer_biases
        )
        hidden_layer_post_activation = relu(hidden_layer_pre_activation)
        output = (
            np.dot(hidden_layer_post_activation, self.output_layer_weights)
            + self.output_layer_biases
        )
        return output
