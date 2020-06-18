import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

    The exact architecture, shown schematically in Fig. 1, is as follows. The input to
    the neural network consists of an 84 x 84 x 4 image produced by the preprocessing map w.

    The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the
    input image and applies a rectifier nonlinearity.
    
    The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again
    followed by a rectifier nonlinearity.

    This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with
    stride 1 followed by a rectifier.
    
    The final hidden layer is fully-connected and consists of 512 rectifier units.
    
    The output layer is a fully-connected linear layer with a single output for
    each valid action.
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        num_actions = self.env.action_space.n
        
        with tf.variable_scope(scope, reuse=reuse):
            h1 = tf.layers.conv2d(state, filters=32, kernel_size=8, strides=4, padding="VALID", activation=tf.nn.relu)
            h2 = tf.layers.conv2d(h1, filters=64, kernel_size=4, strides=2, padding="VALID", activation=tf.nn.relu)
            h3 = tf.layers.conv2d(h2, filters=64, kernel_size=3, strides=1, padding="VALID", activation=tf.nn.relu)
            flattened_h3 = tf.layers.flatten(h3)
            h4 = tf.layers.dense(flattened_h3, 512, activation=tf.nn.relu)
            out = tf.layers.dense(h4, num_actions)

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
