import copy

import numpy as np
import tensorflow as tf

from config import *


class ActorCriticModel(tf.keras.Model):

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
        self.policy_logits = tf.keras.Sequential([
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
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values

    def policy_value_fn(self, board):
        legal_positions = board.availables
        curr_state = np.expand_dims(board.state, axis=0)
        act_probs, value = self(
            tf.convert_to_tensor(curr_state, dtype=tf.float32))
        act_probs = zip(legal_positions, act_probs.numpy()[0][legal_positions])
        return act_probs, value[0]


def mean_policy_value_fn(board):
    availables = board.availables
    action_probs = np.ones(len(availables))/len(availables)
    return zip(availables, action_probs), None


class AlphaZeroA3CError():
    """ A3C Loss 函数 """

    def __call__(self, mcts_probs, policy, rewards, values):
        assert rewards.shape == values.shape
        advantage = tf.squeeze(rewards - values)
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
        policy_loss = tf.keras.losses.categorical_crossentropy(
            y_true=mcts_probs, y_pred=policy)
        assert advantage.shape == policy_loss.shape
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= ENTROPY_BETA * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss


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
