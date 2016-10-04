from .general_nn import GeneralNetworks
import tensorflow as tf

class RecurrentNetworks(GeneralNetworks):

    _implemented_wrappers = {}

    def __init__(self,graph=tf.Graph()):
        super().__init__(graph)




    @staticmethod
    def _dropout_wrapper(cell,input_probability=1.0,output_probability=1.0):
        return tf.nn.rnn_cell.DropoutWrapper(cell,input_probability,output_probability)
    _implemented_wrappers['dropout'] = _dropout_wrapper

    @staticmethod
    def _embedding_wrapper(cell,classes, size, init=None):
        return tf.nn.rnn_cell.EmbeddingWrapper(cell,classes,size,init)
    _implemented_wrappers['embedding'] = _embedding_wrapper

    @staticmethod
    def _input_projection_wrapper(cell,num_proj, input_size=None):
        return tf.nn.rnn_cell.InputProjectionWrapper(cell, num_proj, input_size)
    _implemented_wrappers['input_projection'] = _input_projection_wrapper

    @GeneralNetworks._graphcontext
    def _create_layer(self,l_type,cell_args,wrapper=None,wrapper_args=None):
        with tf.name_scope(l_type+"_"+str(self.depth)):
            if wrapper and not wrapper_args:
                raise ValueError('If using a wrapper, must supply wrapper arguments.')
            if wrapper_args and not wrapper:
                raise ValueError('wrapper_args passed in, but wrapper not set to true')
            self.depth += 1
            cell = GeneralNetworks._implemented['RecurrentNetworks'][l_type].__func__(**cell_args)
            self._nn_structure[self.depth] = cell
            if wrapper:
                return RecurrentNetworks._implemented_wrappers[wrapper].__func__(cell,**wrapper_args)
            return cell


    @staticmethod
    def _basic_rnn(num_units,input_size,activation):
        return ('basic_rnn_cell',tf.nn.rnn_cell.BasicRNNCell(num_units,input_size,activation))

    GeneralNetworks._implemented['RecurrentNetworks']['basic_rnn_cell'] = _basic_rnn

    @staticmethod
    def _basic_lstm(num_units, forget_bias, input_size, activation, state_is_tuple = False):
        return ('basic_lstm_cell',tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias, input_size, state_is_tuple,activation))

    GeneralNetworks._implemented['RecurrentNetworks']['basic_lstm_cell'] = _basic_lstm

    @staticmethod
    def _lstm(num_units, input_size, activation, forget_bias,
            use_peepholes=False, cell_clip=None, initializer=None,
            num_proj=None, proj_clip=None, num_unit_shards=1, num_proj_shards=1,
            state_is_tuple=False):

        args = [num_units, input_size, use_peepholes, cell_clip,
                initializer, num_proj, proj_clip, num_unit_shards,
                num_proj_shards, forget_bias, state_is_tuple, activation]

        return ('full_lstm_cell',tf.nn.rnn_cell.LSTMCell(*args))

    GeneralNetworks._implemented['RecurrentNetworks']['full_lstm_cell'] = _lstm

    @staticmethod
    def _gru(num_units,input_size,activation):
        return ('gru_cell',tf.nn.rnn_cell.GRUCell(num_units,input_size,activation))

    GeneralNetworks._implemented['RecurrentNetworks']['gru_cell'] = _gru

    @staticmethod
    def _multi_cell(cells, state_is_tuple=False):
        return ('multi_cell',tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple))
    
    
    GeneralNetworks._implemented['RecurrentNetworks']['multi_cell'] = _multi_cell
    
    @GeneralNetworks._graphcontext
    def unroll_rnn():
        tf.while()





