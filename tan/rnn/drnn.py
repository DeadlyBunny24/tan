import tensorflow as tf
from tensorflow.python.ops import array_ops

class BasicDRNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a drnn_cell. This cell will activate and update only the neurons
    for a particular timestep. Afterwards, the timestep is updated (i.e. timestep +=1)

    """
    # TODO: Give support for both SRU and FRU
    # TODO: Make cell work without a delay of 1 in the net_arch.
    # TODO: Make architecture work with list of cells

    def __init__(self,
        net_arch=[1],
        rnn_unit=tf.contrib.rnn.BasicRNNCell,
        rnn_unit_params={'unit_size':5}):
        self.net_arch = net_arch
        self.rnn_unit = tf.contrib.rnn.BasicRNNCell
        self.rnn_unit_params=rnn_unit_params
        self.cell={}
        self.timestep=1
        self.state_dim=0

        outputs = []
        for key in net_arch:
            with tf.variable_scope('{}'.format(key),reuse=tf.AUTO_REUSE):
                self.cell[key] = rnn_unit(**rnn_unit_params)
                self.state_dim+=self.cell[key].state_size
                print('Current state size:{}'.format(self.cell[key].state_size))

    @property
    def state_size(self):
        return self.state_dim

    @property
    def output_size(self):
        return self.state_dim

    @property
    def rnn_unit_size(self):
        return self.state_dim/len(self.net_arch)

    @property
    def get_timestep(self):
        return self.timestep

    @property
    def reset_timestep(self):
        self.timestep=1
        print('Dynamic model timestep reset')

    def __call__(self, inputs,state, scope=None):
        cur_state_pos = 0
        output_list=[]
        state_list=[]
        previous_state = inputs
        init_previous_state = inputs

        if tf.get_variable_scope().reuse:
            # This portion activates all cells so that all variables are initialized
            # before OutputProjectionWrapper sets scope.reuse to True.
            # Once it's true, the cells won't be able instantiate new variables.
            # (e.g. kernel, bias)
            for key in self.net_arch:
                cur_state = array_ops.slice(state, [0, cur_state_pos],
                    [-1, self.rnn_unit_size])
                outputs, init_previous_state = self.cell[key](
                    init_previous_state,cur_state,'{}'.format(key))

        for key in self.net_arch:
            cur_state = array_ops.slice(state, [0, cur_state_pos],
                [-1, self.rnn_unit_size])

            if (self.timestep % key)==0:
                output, new_state = self.cell[key](previous_state,cur_state,
                    '{}'.format(key))

            previous_state = new_state
            output_list.append(output)
            state_list.append(new_state)
            cur_state_pos += self.rnn_unit_size
        stacked_outputs = tf.concat(output_list,axis=1)
        stacked_states = tf.concat(state_list,axis=1)
        self.timestep+=1
        return stacked_outputs,stacked_states

    def zero_state(self,batch_size,dtype):
        return tf.zeros([batch_size,self.state_dim],
            dtype=dtype)
