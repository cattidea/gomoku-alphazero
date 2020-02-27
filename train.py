import multiprocessing
import os
import queue
import threading
import time
import argparse
import random

import numpy as np
import tensorflow as tf

from policy import ActorCriticModel, AlphaZeroA3CError, AlphaZeroError
from config import *
from board import Board
from play import Game, MCTSA3CPlayer, MCTSPlayer
from ui import HeadlessUI

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Gomoku A3C')
parser.add_argument('--resume', action='store_true', help='恢复模型继续训练')
args = parser.parse_args()


class DataAugmentor():
    """ 数据扩增器
    对原数据进行旋转 + 对称，共八种扩增方式 """
    def __init__(self, rotate=True, flip=True):
        self.rotate = rotate
        self.flip = flip

    def __call__(self, data_batch):
        data_batch_aug = []
        for state, mcts_prob, reward in data_batch:
            state_aug = state
            mcts_prob_aug = mcts_prob.reshape(WIDTH, HEIGHT)
            if self.rotate:
                num_revo = np.random.randint(4)
                state_aug = np.rot90(state_aug, num_revo)
                mcts_prob_aug = np.rot90(mcts_prob_aug, num_revo)
            if self.flip and np.random.random() > 0.5:
                state_aug = np.fliplr(state_aug)
                mcts_prob_aug = np.fliplr(mcts_prob_aug)
            mcts_prob_aug = mcts_prob_aug.flatten()
            data_batch_aug.append((state_aug, mcts_prob_aug, reward))
        return data_batch_aug


class AlphaZeroMetric():
    """ AlphaZero 性能评估器 """

    def __init__(self, n_playout=400):
        self.n_playout = n_playout
        self.n_playout_mcts = 1000
        self.best_score = 0.

    def __call__(self, weights, episode=0, n_games=10):
        assert n_games % 2 == 0

        mcts_a3c_player = MCTSA3CPlayer(c_puct=5, n_playout=self.n_playout)
        mcts_a3c_player.model.build(input_shape=(None, WIDTH, HEIGHT, CHANNELS))
        mcts_a3c_player.model.set_weights(weights)
        mcts_player = MCTSPlayer(c_puct=5, n_playout=self.n_playout_mcts)
        game = Game(mcts_a3c_player, mcts_player, HeadlessUI())
        scores = {WIN: 0, LOSS: 0, TIE: 0}
        score = 0.
        for _ in range(n_games):
            winner = game.play(is_selfplay=False)
            scores[winner * mcts_a3c_player.color] += 1
            game.switch_players()
        for key in scores:
            score += key * scores[key]
        print('[Test] Episode: {:5d}, MCTS n_playout: {:6d}, Win: {:2d}, Loss: {:2d}, Tie: {:2d}, Score: {:.2f} '.format(
            episode, self.n_playout_mcts, scores[WIN], scores[LOSS], scores[TIE], score
        ))
        if score > self.best_score:
            self.best_score = score
            if score == n_games:
                self.best_score = 0.
                self.n_playout_mcts += 500
            return True
        return False


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    save_lock = threading.Lock()
    metric = AlphaZeroMetric(n_playout=400)

    def __init__(self,
                 global_model,
                 opt,
                 result_queue,
                 idx):
        super().__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.worker_idx = idx
        self.loss_object = AlphaZeroError()
        self.mean_loss = tf.keras.metrics.Mean(name='train_loss')
        self.player = MCTSA3CPlayer(c_puct=5, n_playout=400)
        self.local_model = self.player.model
        self.local_model.build(input_shape=(None, WIDTH, HEIGHT, CHANNELS))
        self.local_model.set_weights(self.global_model.get_weights())
        self.game = Game(self.player, self.player, HeadlessUI())
        self.data_aug = DataAugmentor(rotate=True, flip=True)

    def run(self):
        while Worker.global_episode < MAX_EPISODE:
            episode = Worker.global_episode
            winner = self.game.play(is_selfplay=True)

            total_loss = tf.constant(0)

            for epoch in range(EPOCHS):
                mini_batch = random.sample(self.game.data_buffer, min(BATCH_SIZE, len(self.game.data_buffer)//2))
                mini_batch = self.data_aug(mini_batch)
                states_batch = tf.convert_to_tensor([data[0] for data in mini_batch], dtype=tf.float32)
                mcts_probs_batch = tf.convert_to_tensor([data[1] for data in mini_batch], dtype=tf.float32)
                rewards_batch = tf.convert_to_tensor(np.expand_dims([data[2] for data in mini_batch], axis=-1), dtype=tf.float32)

                self.local_model.set_weights(self.global_model.get_weights())

                with tf.GradientTape() as tape:
                    policy, values = self.local_model(states_batch)

                    total_loss = self.loss_object(
                        mcts_probs=mcts_probs_batch,
                        policy=policy,
                        rewards=rewards_batch,
                        values=values)

                # Calculate local gradients
                grads = tape.gradient(
                    total_loss, self.local_model.trainable_weights)
                # Push local gradients to global model
                self.opt.apply_gradients(
                    zip(grads, self.global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(self.global_model.get_weights())

                self.mean_loss(total_loss)

                print('[Training] Episode: {:5d}, Epoch: {:2d}, Worker: {:2d}, Winner: {:5s}, Loss: {}  '.format(
                    episode+1,
                    epoch+1,
                    self.worker_idx,
                    COLOR[winner],
                    total_loss.numpy()), end='\r')

            if (episode + 1) % CHECK_FREQ == 0:
                print('[Training] Episode: {:5d}, Loss: {}  '.format(
                    episode+1,
                    self.mean_loss.result()
                ))
                self.mean_loss.reset_states()
                is_best_score = Worker.metric(self.global_model.get_weights(), episode)
                with Worker.save_lock:
                    if is_best_score:
                        self.global_model.save_weights(MODEL_FILE)

            self.result_queue.put(total_loss.numpy())
            with Worker.save_lock:
                Worker.global_episode += 1
        self.result_queue.put(None)


def train():
    global_model = ActorCriticModel()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    global_model.build(input_shape=(None, WIDTH, HEIGHT, CHANNELS))
    global_model.summary()
    if args.resume:
        global_model.load_weights(MODEL_FILE)

    res_queue = queue.Queue()

    workers = [Worker(global_model,
                      optimizer,
                      res_queue,
                      i) for i in range(multiprocessing.cpu_count())]

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
    [w.join() for w in workers]


if __name__ == '__main__':
    train()
