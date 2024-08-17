import numpy as np


# TODO: this can also be applied to arbitrary-dimensional tensors, but might produce an unexpected result, softmaxing
# the whole tensor (to add to 1) rather than applying softmax to each row. Does that need fixing? Probably either
# rename `vec` to clarify that any tensor would work, or error if the dimensions aren't 1
def softmax(vec):
    sum_of_exps = np.sum(np.exp(vec))
    return np.exp(vec) / sum_of_exps


def relu(x):
    return np.maximum(0, x)


def xavier_initialize(in_dimension, out_dimension):
    limit = np.sqrt(6 / (in_dimension + out_dimension))
    return np.random.uniform(-limit, limit, (in_dimension, out_dimension))


"""
"Even if our efforts of attention seem for years to be producing no result, one day a light that is in exact proportion to them will flood the soul."
"""