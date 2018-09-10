import tensorflow as tf
from tensorflow.python.ops import array_ops

class BasicDRNNCell(tf.contrib.rnn.RNNCell):

    # TODO: Do parameter checking.
    # TODO: Give support for SRU and FRU.
    # TODO: Make cell work without a delay of 1 in the net_arch.
    # TODO: Provide multi-unit support.
    # TODO: Provide multiple equal timesteps in net_arch.
    # TODO: Provide scope and name support.

    def __init__(self,rnn_unit_size,net_arch=[1],
        rnn_unit=tf.contrib.rnn.BasicRNNCell,activation=tf.nn.tanh):
        """
        Instantiates a 'drnn_cell'. Every instance has a 'timestep',
        this attribute control's the activation of the recurrent units. To reset
        'timestep' use reset_timestep().

        Args:
          rnn_unit_size: Size of each recurrent unit.
          net_arch: List of timesteps to consider. Every new timestep will
          instantiate a new recurrent unit.
          rnn_unit: Class of recurrent units.
          activation: Activation function of the recurrent units.

        """
        self.rnn_unit_size = rnn_unit_size
        self.net_arch = net_arch
        self.activation = activation
        self.cell={}
        self.timestep=1
        outputs = []
        for key in net_arch:
            with tf.variable_scope('{}'.format(key),reuse=tf.AUTO_REUSE):
                self.cell[key] = rnn_unit(rnn_unit_size)

    @property
    def state_size(self):
        return len(self.net_arch)*self.rnn_unit_size

    @property
    def output_size(self):
        return len(self.net_arch)*self.rnn_unit_size

    @property
    def get_timestep(self):
        return self.timestep

    @property
    def reset_timestep(self):
        self.timestep=1
        print('Dynamic model timestep reset')

    @property
    def zero_state(self,batch_size,dtype):
        return tf.zeros([batch_size,len(self.net_arch)*self.rnn_unit_size],
            dtype=dtype)

    def __call__(self, inputs, state, scope=None):
        """
        Runs the cells that activate on current timestep value. Input is fed to
        the first cell.
        """
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
        return tf.zeros([batch_size,len(self.net_arch)*self.rnn_unit_size],
            dtype=dtype)
