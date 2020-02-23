import multiprocessing
import os
import queue
import threading
import time
import argparse
import random

import numpy as np
import tensorflow as tf

from policy import ActorCriticModel
from config import *
from board import Board
from play import Game, MCTSA3CPlayer, MCTSPlayer
from ui import HeadlessUI

'''
based on: https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Gomoku A3C')
parser.add_argument('--resume', action='store_true', help='恢复模型继续训练')
args = parser.parse_args()


class Memory:
    def __init__(self):
        self.color = 0
        self.states = []
        self.actions = []
        self.reward = 0
        self.cnt = 0

    def store(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        self.cnt += 1

    def set_reward(self, reward):
        self.reward = reward

    def clear(self):
        self.color = 0
        self.states.clear()
        self.actions.clear()
        self.reward = 0
        self.cnt = 0


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    save_lock = threading.Lock()

    def __init__(self,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 gamma=0.99,
                 ):
        super().__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.worker_idx = idx
        self.gamma = gamma
        self.mean_loss = tf.keras.metrics.Mean(name='train_loss')
        self.player = MCTSA3CPlayer(c_puct=5, n_playout=400)
        self.local_model = self.player.model
        self.local_model.build(input_shape=(None, width, height, 4))
        self.local_model.set_weights(self.global_model.get_weights())
        self.game = Game(self.player, self.player, HeadlessUI())

    def run(self):
        while Worker.global_episode < MAX_EPISODE:
            episode = Worker.global_episode
            winner = self.game.play(is_selfplay=True)
            total_loss = tf.constant(0.0)
            if len(self.game.data_buffer) >= BATCH_SIZE:
                mini_batch = random.sample(self.game.data_buffer, BATCH_SIZE)
                states_batch = [data[0] for data in mini_batch]
                mcts_probs_batch = [data[1] for data in mini_batch]
                rewards_batch = [data[2] for data in mini_batch]

                with tf.GradientTape() as tape:
                    policy, values = self.local_model(tf.convert_to_tensor(
                        np.array(states_batch), dtype=tf.float32))

                    # Get our advantages
                    advantage = tf.convert_to_tensor(np.array(rewards_batch)[:, None],
                                                     dtype=tf.float32) - values

                    # Value loss
                    value_loss = advantage ** 2

                    # Calculate our policy loss
                    entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
                    policy_loss = tf.keras.losses.categorical_crossentropy(
                        y_true=mcts_probs_batch, y_pred=policy)
                    policy_loss *= tf.stop_gradient(advantage)
                    policy_loss -= ENTROPY_BETA * entropy
                    total_loss = tf.reduce_mean(
                        (0.5 * value_loss + policy_loss))

                # Calculate local gradients
                grads = tape.gradient(
                    total_loss, self.local_model.trainable_weights)
                # Push local gradients to global model
                self.opt.apply_gradients(
                    zip(grads, self.global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(self.global_model.get_weights())

                self.mean_loss(total_loss)
                print('Episode: {:5d}, Worker: {:2d}, Winner: {:5s}, Loss: {}  '.format(
                    episode+1,
                    self.worker_idx,
                    COLOR[winner],
                    total_loss.numpy()), end='\r')

            if (episode + 1) % 50 == 0:
                print('Episode: {:5d}, Loss: {}  '.format(
                    episode+1,
                    self.mean_loss.result()
                ))
                self.mean_loss.reset_states()
                with Worker.save_lock:
                    self.global_model.save_weights(MODEL_FILE)

            self.result_queue.put(total_loss.numpy())
            with Worker.save_lock:
                Worker.global_episode += 1
        self.result_queue.put(None)


def train():
    global_model = ActorCriticModel()
    learning_rate = 1e-4
    gamma = 0.99
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    global_model.build(input_shape=(None, width, height, 4))
    if args.resume:
        global_model.load_weights(MODEL_FILE)

    res_queue = queue.Queue()

    workers = [Worker(global_model,
                      optimizer,
                      res_queue,
                      i,
                      gamma=gamma) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
        print("Starting worker {}".format(i))
        worker.setDaemon(True)
        worker.start()

    while True:
        try:
            reward = res_queue.get()
            if reward is None:
                break
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    # [w.join() for w in workers]


if __name__ == '__main__':
    train()
