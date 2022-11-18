# model creation and training
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        self.sess, self.saver = self.setup_tf()

    def setup_cnn(self):
        pass

    def setup_rnn(self):
        pass

    def setup_ctc():
        pass