import tensorflow as tf
from tensorflow.python.ops import array_ops

class BasicDRNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a drnn_cell. This cell will activate and update only the neurons
    for a particular timestep. Afterwards, the timestep is updated (i.e. timestep +=1)
    Make sure to reset timestep when the sequence finishes.
    """
    # TODO: Make cell work without a delay of 1 in the net_arch.
    # TODO: Make architecture work with list of cells

    def __init__(self,
        net_arch=[1],
        cell={},
        hidden_size=0,
        state_time_tuple=False):
        self.net_arch = net_arch
        self.cell=cell
        self.timestep=1
        self.hidden_size=hidden_size
        self.state_time_tuple=state_time_tuple

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def rnn_unit_size(self):
        return self.hidden_size/len(self.net_arch)

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
                if (self.state_time_tuple):
                    cur_state = (cur_state,self.timestep)
                outputs, init_previous_state = self.cell[key](
                    init_previous_state,cur_state,'{}'.format(key))

        for key in self.net_arch:
            cur_state = (array_ops.slice(state, [0, cur_state_pos],
                [-1, self.rnn_unit_size]))

            if self.state_time_tuple:
                cur_state = (cur_state,self.timestep)

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
        return tf.zeros([batch_size,self.hidden_size],
            dtype=dtype)
