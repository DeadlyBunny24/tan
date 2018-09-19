import tensorflow as tf
from tensorflow.python.ops import array_ops

class BasicDRNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a drnn_cell. This cell will activate and update only the neurons
    for a particular timestep. Afterwards, the timestep is updated (i.e. timestep +=1)
    Make sure to reset timestep when the sequence finishes.
    """
    # TODO: Make cell work without a delay of 1 in the net_arch.
    # TODO: Make architecture work with layers of different cells.
    # TODO: Clean up cell initialization
    # TODO: Remove cur_list_key from the cell attributes. It is kept to
    # visualize better in Tensorboard.

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
        with tf.name_scope('t_{}'.format(self.timestep)):
            cur_state_pos = 0
            output_list=[]
            init_previous_state = inputs
            cur_list_key={}

            # This loop extracts the last state of each recurrent unit.
            for key in self.net_arch:
                cur_state = array_ops.slice(state, [0, cur_state_pos],
                    [-1, self.rnn_unit_size])
                cur_list_key[key]=cur_state
                print('current state pos:{}'.format(cur_state_pos))
                cur_state_pos+=self.rnn_unit_size

            if not tf.get_variable_scope().reuse:
                # This portion activates all cells so that all variables are initialized
                # before OutputProjectionWrapper sets scope.reuse to True.
                # Once it's true, the cells won't be able instantiate new variables.
                # (e.g. kernel, bias)
                print('Initializing cells')
                for key in self.net_arch:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                        [-1, self.rnn_unit_size])
                    if (self.state_time_tuple):
                        cur_state = (cur_state,self.timestep)
                    outputs, init_previous_state = self.cell[key](
                        init_previous_state,cur_state,'{}'.format(key))
                # return state,state

            # This sets the reuse of the previously initialized variables to True.
            # That way, TensorFlow will look for the variables instead of creating them
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                for key in self.net_arch:
                    if self.state_time_tuple:
                        cur_state = (cur_list_key[key],self.timestep)
                    else:
                        cur_state =  cur_list_key[key]
                    if key==1:
                        with tf.name_scope('{}'.format(key)):
                            output, cur_list_key[key] = self.cell[key](inputs,
                            cur_state,'{}'.format(key))
                    elif (self.timestep % key)==0:
                        with tf.name_scope('{}'.format(key)):
                            output, cur_list_key[key] = self.cell[key](
                            cur_list_key[previous_key],cur_state,'{}'.format(key))
                    # This works because the first key sets the value of previous_key
                    # A recurrent unit with delay one is nececesary for this to work.
                    previous_key = key
                    output_list.append(output)
                stacked_outputs = tf.concat(output_list,axis=1)
                stacked_states = tf.concat([cur_list_key[key] \
                for key in self.net_arch],axis=1)
                self.timestep+=1
                return stacked_outputs,stacked_states

    def zero_state(self,batch_size,dtype):
        return tf.zeros([batch_size,self.hidden_size],
            dtype=dtype)
