import copy

import numpy as np
import tensorflow as tf

from config import *


class PolicyValueModelBase(tf.keras.Model):

    def policy_value_fn(self, board):
        legal_positions = board.availables
        curr_state = np.expand_dims(board.state, axis=0)
        act_probs, value = self(
            tf.convert_to_tensor(curr_state, dtype=tf.float32))
        act_probs = zip(legal_positions, act_probs.numpy()[0][legal_positions])
        return act_probs, value[0]


class PolicyValueModel(PolicyValueModelBase):

    def __init__(self):
        super().__init__()
        l2_const = 1e-4
        self.base_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, strides=1, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Conv2D(
                64, 3, strides=1, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Conv2D(
                128, 3, strides=1, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_const))
        ])
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 1, strides=1, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(WIDTH*HEIGHT, kernel_regularizer=tf.keras.regularizers.l2(l2_const),
                                  activation='softmax')
        ])
        self.values = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 1, strides=1, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                64, kernel_regularizer=tf.keras.regularizers.l2(l2_const)),
            tf.keras.layers.Dense(
                1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2_const))
        ])

    @tf.function
    def call(self, inputs):
        x = self.base_net(inputs)
        policy = self.policy(x)
        values = self.values(x)
        return policy, values


class PolicyValueModelResNet(PolicyValueModelBase):

    def __init__(self):
        super().__init__()
        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU()
        ])
        self.res_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        ])
        self.res_2 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        ])
        self.res_3 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        ])
        self.postprocess = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
        ])
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(WIDTH*HEIGHT, activation='softmax')
        ])
        self.values = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(1, activation='tanh')
        ])

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.preprocess(x)
        x = tf.keras.layers.add([x, self.res_1(x)])
        x = tf.keras.layers.add([x, self.res_2(x)])
        x = tf.keras.layers.add([x, self.res_3(x)])
        x = self.postprocess(x)
        policy = self.policy(x)
        values = self.values(x)
        return policy, values


def mean_policy_value_fn(board):
    availables = board.availables
    action_probs = np.ones(len(availables))/len(availables)
    return zip(availables, action_probs), None


class AlphaZeroError():
    """ AlphaZero Loss 函数 """

    def __call__(self, mcts_probs, policy, rewards, values):

        assert rewards.shape == values.shape

        policy_loss = tf.keras.losses.categorical_crossentropy(
            y_true=mcts_probs, y_pred=policy)
        value_loss = tf.keras.losses.MSE(
            y_true=rewards, y_pred=values)
        total_loss = tf.reduce_mean(value_loss + policy_loss)

        return total_loss
