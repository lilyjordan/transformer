import numpy as np


def softmax(vec):
    sum_of_exps = np.sum(np.exp(vec))
    return np.exp(vec) / sum_of_exps