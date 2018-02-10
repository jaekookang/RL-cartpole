import tensorflow as tf
import numpy as np


class DeepQNet():
    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str='main') -> None:
        '''
        (1) Setup Neural Network
        (2) Predict Q vector given current state
        (3) Train the Neural Network
        '''
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._setup_network()

    def _setup_network(self, hid_layer=[16], learn_rate=0.001) -> None:
        '''
        Set up a MLP structure
        '''
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(
                tf.float32, [None, self.input_size], name='input_x')
            net = self._X

            for nhid in hid_layer:
                net = tf.layers.dense(net, nhid, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.output_size)
            self._Qpred = net

            self._Y = tf.placeholder(tf.float32, [None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        '''
        Returns Q(s, a) = [r1, r2, ... rN]; (N, output_size)
        '''
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, {self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        '''
        Update Q function (=MLP)
        given minibatch of x and y
        '''
        feed = {
            self._X: x_stack,
            self._Y: y_stack,
        }
        return self.session.run([self._loss, self._train], feed)
