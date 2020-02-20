import copy
import queue

import numpy as np
import tensorflow as tf

from config import BLACK, WHITE, height, width

POINT_QUEUE = queue.Queue()


class ActorCriticModel(tf.keras.Model):
    """
    ref: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/policy_value_net_keras.py
    """

    def __init__(self):
        super().__init__()
        l2_const = 1e-4
        self.base_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                64, 3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                128, 3, strides=1, padding='same', activation='relu')
        ])
        self.policy_logits = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 1, strides=1, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(width*height)
        ])
        self.values = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 1, strides=1, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                64, kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Dense(
                1, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l2_const))
        ])

    def call(self, inputs):
        x = self.base_net(inputs)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values


class Player():

    def __init__(self):
        pass

    def __call__(self, env):
        raise NotImplementedError

    def set_reward(self, reward):
        pass


class Computer(Player):

    def __init__(self, weights=None, prob=False):
        super().__init__()
        self.model = ActorCriticModel()
        self.prob = prob
        if weights is not None:
            self.model.build(input_shape=(None, width, height, 1))
            self.model.load_weights(weights)

    def __call__(self, curr_state):
        curr_state = np.expand_dims(curr_state, axis=-1)
        curr_state = np.expand_dims(curr_state, axis=0)

        logits, _ = self.model(tf.convert_to_tensor(
            curr_state, dtype=tf.float32))
        mask = tf.reshape(curr_state == 0, (width*height, ))
        logits = tf.where(mask, logits, -np.inf)
        probs = tf.nn.softmax(logits)

        action = np.random.choice(
            height*width, p=probs.numpy()[0]) if self.prob else np.argmax(probs)
        x, y = action // width, action % width
        return x, y


class Human(Player):

    def __init__(self):
        super().__init__()

    def __call__(self, env):
        POINT_QUEUE.queue.clear()
        while True:
            x, y = POINT_QUEUE.get()
            if x >= 0 and x < width and y >= 0 and y < height and env[x, y] == 0:
                return x, y
