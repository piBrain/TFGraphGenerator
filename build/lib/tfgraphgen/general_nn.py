import tensorflow as tf
from collections import OrderedDict
import math


class GeneralNetworks():

    _implemented = {
                'GeneralNetworks': {},
                'RecurrentNetworks': {}
            }



    def __init__(self,graph=tf.Graph()):
        self._nn_structure = OrderedDict()
        self.build_started = False
        self.depth = 0
        self.tf_graph = graph

    @staticmethod
    def get_tensorflow():
        return tf

    def start_build(self,label,zeroeth_layer):
        self.build_started = True
        self._nn_structure[0] = (label,zeroeth_layer)
        return  self._layer_gen()

    def __repr__(self):


        structure = self._nn_structure

        _repr = ""

        for k in structure:

            if k == 0 and k != len(structure)-1 :
                _repr+="Layer: "+str(k)+" Type: "+str(structure[k])+"---"
                continue
            if k == (len(structure)-1):
                _repr+="--->Layer: "+str(k)+" Type: "+str(structure[k])
                continue
            _repr+="--->Layer: "+str(k)+" Type: "+str(structure[k])+"---"

        return _repr

    def _graphcontext(f):
        def wrap(self,*args,**kwargs):
            with self.tf_graph.as_default():
               f(self,*args,**kwargs)
        return wrap

    @staticmethod
    def _check_l_type(self,l_type):
        if l_type not in GeneralNetworks._implemented[self.__class__.__name__].keys():
            raise ValueError("l_type not implemented")

    def _layer_gen(self):
        while True:
            parameters = yield
            try:
                l_type,args = parameters
                GeneralNetworks._check_l_type(l_type)
                self._create_layer(l_type,*args)
                yield self._nn_structure[self.depth]
            except TypeError as e:
                raise e

    @_graphcontext
    def _create_layer(self,l_type,weights,biases,kwargs):
        with tf.name_scope(l_type+"_"+str(self.depth)):
            self.depth += 1
            tf_weights = tf.Variable(
                    tf.truncated_normal([weights, biases],
                    stddev=1.0 / math.sqrt(float(weights))),name='weights')
            tf_biases = tf.Variable(tf.zeros([biases]),name = 'biases')

            previous_layer_output = self._nn_structure[self.depth-1][1]
            self._nn_structure[self.depth] = GeneralNetworks._implemented['GeneralNetworks'][l_type].__func__(previous_layer_output,tf_weights,tf_biases,**kwargs)

    @staticmethod
    def _relu(inputs,weights,biases):
        return ('relu',tf.nn.relu(tf.matmul(inputs,weights)+biases))
    _implemented['GeneralNetworks']["relu_layer"] = _relu

    @staticmethod
    def _softmax(inputs,weights,biases):
        return ('soft_max',tf.matmul(inputs,weights)+biases)
    _implemented['GeneralNetworks']["softmax_layer"] = _softmax


    @staticmethod
    def _leaky_relu(inputs, weights, biases, alpha = 0.1):
        linear_system = tf.matmul(inputs,weights)+biases
        leaky_max = tf.maximum(alpha*linear_system,linear_system)
        return ('leaky_relu', leaky_max)
    _implemented['GeneralNetworks']['leaky_relu_layer'] = _leaky_relu

    @staticmethod
    def _tanh(inputs, weights, biases):
        return ('tanh',tf.nn.tanh(tf.matmul(inputs,weights)+biases))
    _implemented['GeneralNetworks']['tanh_layer'] = _tanh
    @staticmethod
    def _elu(inputs, weights, biases):
        return ('elu',tf.nn.elu(tf.matmul(inputs,weights)+biases))
    _implemented['GeneralNetworks']['elu_layer'] = _elu

    @staticmethod
    def _log_sigmoid(inputs, weights, biases):
        return ('log_sigmoid', tf.nn.sigmoid(tf.matmul(inputs,weights)+biases))
    _implemented['GeneralNetworks']['sigmoid_layer'] = _log_sigmoid
    @staticmethod
    def _softsign(inputs,weights,biases):
        return ('softsign',tf.nn.softsign(tf.matmul(inputs,weights)+biases))
    _implemented['GeneralNetworks']['softsign_layer'] = _softsign
    @staticmethod
    def _dropout(inputs,weights,biases,keep,noise_shape=None):
        return ('dropout',tf.nn.dropout(tf.matmul(inputs,weights,biases),keep,noise_shape))
    _implemented['GeneralNetworks']['dropout_layer'] = _dropout
    @staticmethod
    def _relu6(inputs,weights,biases):
        return ('relu6',tf.nn.relu6(tf.matmul(inputs,weights,biases)))
    _implemented['GeneralNetworks']['relu6_layer'] = _relu6
    @staticmethod
    def _softplus(inputs,weights,biases):
        return ('softplus',tf.nn.softplus(tf.matmul(inputs,weights,biases)))
    _implemented['GeneralNetworks']['softplus_layer'] = _softplus



