import copy

import numpy as np
import tensorflow as tf

from config import BLACK, WHITE, height, width


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
            tf.keras.layers.Dense(width*height, activation='softmax')
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

    def call(self, inputs):
        x = self.base_net(inputs)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values

    def policy_value_fn(self, board):
        legal_positions = board.availables
        curr_state = np.expand_dims(board.state, axis=0)
        act_probs, value = self(tf.convert_to_tensor(curr_state, dtype=tf.float32))
        act_probs = zip(legal_positions, act_probs.numpy()[0][legal_positions])
        return act_probs, value[0]


def mean_policy_value_fn(board):
    availables = board.availables
    action_probs = np.ones(len(availables))/len(availables)
    return zip(availables, action_probs), None
