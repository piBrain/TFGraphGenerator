class nn_graph_gen():
    
    __implemented_activations = {}

   
    
    def __init__(self,inputs,graph):

        assert isinstance(inputs,tf.Variable) or isinstance(inputs,tf.Tensor)
        self.__nn_structure = OrderedDict()
        self.__nn_structure[0] = ("input",inputs)
        self.depth = 0
        self.tf_graph = graph
        
    def __repr__(self):


        structure = self.__nn_structure

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
            print(self.tf_graph == graph)
            with self.tf_graph.as_default():
                f(self,*args,**kwargs)
        return wrap
    

    def layer_gen(self):

        while True:
            parameters = yield
            try:
                l_type,in_1,in_2 = parameters
                assert l_type in nn_graph_gen.__implemented_activations.keys()
                self._create_layer(l_type,in_1,in_2)
                yield self.__nn_structure[self.depth]
            except TypeError as e:
                raise e  
                
    @_graphcontext
    def _create_layer(self,l_type,in_1,in_2):
        with tf.name_scope(l_type+"_"+str(self.depth)):
            self.depth += 1
            weights = tf.Variable(
                    tf.truncated_normal([in_1, in_2],
                    stddev=1.0 / math.sqrt(float(in_1))),name='weights')
            biases = tf.Variable(tf.zeros([in_2]),name = 'biases')

            previous_layer_output = self.__nn_structure[self.depth-1][1]
            self.__nn_structure[self.depth] = nn_graph_gen.__implemented_activations[l_type].__func__(previous_layer_output,weights,biases)
   
    @staticmethod
    def _relu(inputs,weights,biases):
        return ('relu',tf.nn.relu(tf.matmul(inputs,weights)+biases))
    __implemented_activations["relu_layer"] = _relu  
   
    @staticmethod
    def _softmax(inputs,weights,biases):
        return ('soft_max',tf.matmul(inputs,weights)+biases)
    __implemented_activations["softmax_layer"] = _softmax