from .general_nn import GeneralNetworks
import tensorflow as tf

class RecurrentNetworks(GeneralNetworks):

    _implemented_wrappers = {}

    def __init__(self,graph=tf.Graph()):
        super().__init__(graph)

    def _wrapper_assertions(self,wrapper,wrapper_args):
        if wrapper and not wrapper_args:
            raise ValueError('If using a wrapper, must supply wrapper arguments.')
        if wrapper_args and not wrapper:
            raise ValueError('wrapper_args passed in, but wrapper not set to true')


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
            self._wrapper_assertions(wrapper,wrapper_args)
            self.depth += 1
            cell = GeneralNetworks._implemented['RecurrentNetworks'][l_type].__func__(**cell_args)
            self._nn_structure[self.depth] = cell
            if wrapper:
                return RecurrentNetworks._implemented_wrappers[wrapper].__func__(cell,**wrapper_args)
            return cell

    @GeneralNetworks._graphcontext
    def chain(self,new_l_type,dyn_rnn_kwargs,next_layer_kwargs,state_saving=False,wrapper=None,wrapper_args=None):
        if state_saving:
            output = tf.nn.state_saving_rnn(self._nn_structure[self.depth][1],**dyn_rnn_kwargs)
        else:
            output = tf.nn.dynamic_rnn(self._nn_structure[self.depth][1],**dyn_rnn_kwargs)
        self._wrapper_assertions(wrapper,wrapper_args)
        self._check_l_type(new_l_type)
        self.depth += 1
        self._nn_structure[self.depth]=('OUTPUT_LAYER',)
        self.create_layer(new_l_type,next_layer_kwargs,wrapper,wrapper_args)
        return output


    @GeneralNetworks._graphcontext
    def output(self,dyn_rnn_kwargs,state_saving=False,wrapper=None,wrapper_args=None,short_out_cell=None):
        print(self.depth)
        if self.depth == 1:
            if short_out_cell is None:
                raise(ValueError,'Network Depth is 0 and short_out_cell is None.')
            self._check_l_type(short_out_cell[0])
            cell = GeneralNetworks._implemented['RecurrentNetworks'][short_out_cell[0]].__func__(**short_out_cell[1])[1]
            output = tf.nn.dynamic_rnn(cell,**dyn_rnn_kwargs)
        elif state_saving:
            output = tf.nn.state_saving_rnn(self._nn_structure[self.depth][1],**dyn_rnn_kwargs)
        else:
            output = tf.nn.dynamic_rnn(self._nn_structure[self.depth][1],**dyn_rnn_kwargs)
        self._wrapper_assertions(wrapper,wrapper_args)
        self.depth += 1
        self._nn_structure[self.depth]=('OUTPUT_LAYER',)
        return output

    @staticmethod
    def _basic_rnn(num_units,input_size,activation):
        return ('basic_rnn_cell',tf.nn.rnn_cell.BasicRNNCell(num_units,input_size,activation))

    GeneralNetworks._implemented['RecurrentNetworks']['basic_rnn_cell'] = _basic_rnn

    @staticmethod
    def _basic_lstm(num_units, forget_bias, activation, state_is_tuple = False):
        return ('basic_lstm_cell',tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias, input_size=None, state_is_tuple=state_is_tuple,activation=activation))

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






