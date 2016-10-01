import tensorflow as tf
from collections import OrderedDict
import math

class RecurrentNetworks(GeneralNetworks):

    @staticmethod
    def _basic_rnn_layer(inputs,weights,biases):
        return ('relu',tf.nn.relu(tf.matmul(inputs,weights)+biases))
    __implemented_activations["relu_layer"] = _relu





