import os
import queue
import time
import argparse
import random

import numpy as np
import tensorflow as tf

from policy import PolicyValueModel, AlphaZeroError
from config import *
from board import Board
from play import Game, MCTSAlphaZeroPlayer, MCTSPlayer
from ui import HeadlessUI


parser = argparse.ArgumentParser(description='Gomoku AlphaZero')
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
        self.best_score = -np.inf

    def __call__(self, weights, episode=0, n_games=10):
        assert n_games % 2 == 0

        mcts_alphazero_player = MCTSAlphaZeroPlayer(
            c_puct=5, n_playout=self.n_playout)
        mcts_alphazero_player.model.build(
            input_shape=(None, WIDTH, HEIGHT, CHANNELS))
        mcts_alphazero_player.model.set_weights(weights)
        mcts_player = MCTSPlayer(c_puct=5, n_playout=self.n_playout_mcts)
        game = Game(mcts_alphazero_player, mcts_player, HeadlessUI())
        scores = {WIN: 0, LOSS: 0, TIE: 0}
        score = 0.
        for idx in range(n_games):
            winner = game.play(is_selfplay=False)
            res = winner * mcts_alphazero_player.color
            scores[res] += 1
            game.switch_players()
            print('[Testing] Episode: {:5d}, Game: {:2d}, Score: {:2d} '.format(
                episode + 1, idx, res
            ), end='\r')
        for key in scores:
            score += key * scores[key]
        print('[Test] Episode: {:5d}, MCTS n_playout: {:6d}, Win: {:2d}, Loss: {:2d}, Tie: {:2d}, Score: {:.2f} '.format(
            episode + 1, self.n_playout_mcts, scores[WIN], scores[LOSS], scores[TIE], score
        ))
        if score > self.best_score:
            self.best_score = score
            if score == n_games:
                self.best_score = -np.inf
                self.n_playout_mcts += 500
            return True
        return False


class Worker():

    def __init__(self):
        self.player = MCTSAlphaZeroPlayer(c_puct=5, n_playout=400)
        self.model = self.player.model
        self.model.build(input_shape=(None, WIDTH, HEIGHT, CHANNELS))
        self.model.summary()
        if args.resume:
            self.model.load_weights(MODEL_FILE)

        self.opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.loss_object = AlphaZeroError()
        self.mean_loss = tf.keras.metrics.Mean(name='train_loss')
        self.game = Game(self.player, self.player, HeadlessUI())
        self.data_aug = DataAugmentor(rotate=True, flip=True)
        self.metric = AlphaZeroMetric(n_playout=400)

    def run(self):
        for episode in range(MAX_EPISODE):
            winner = self.game.play(is_selfplay=True)

            total_loss = tf.constant(0)

            for epoch in range(EPOCHS):
                mini_batch = random.sample(self.game.data_buffer, min(
                    BATCH_SIZE, len(self.game.data_buffer)//2))
                mini_batch = self.data_aug(mini_batch)
                states_batch = tf.convert_to_tensor(
                    [data[0] for data in mini_batch], dtype=tf.float32)
                mcts_probs_batch = tf.convert_to_tensor(
                    [data[1] for data in mini_batch], dtype=tf.float32)
                rewards_batch = tf.convert_to_tensor(np.expand_dims(
                    [data[2] for data in mini_batch], axis=-1), dtype=tf.float32)

                with tf.GradientTape() as tape:
                    policy, values = self.model(states_batch)

                    total_loss = self.loss_object(
                        mcts_probs=mcts_probs_batch,
                        policy=policy,
                        rewards=rewards_batch,
                        values=values)

                grads = tape.gradient(
                    total_loss, self.model.trainable_weights)
                self.opt.apply_gradients(
                    zip(grads, self.model.trainable_weights))

                self.mean_loss(total_loss)

                print('[Training] Episode: {:5d}, Epoch: {:2d}, Winner: {:5s}, Loss: {}  '.format(
                    episode+1,
                    epoch+1,
                    COLOR[winner],
                    total_loss.numpy()), end='\r')

            if (episode + 1) % CHECK_FREQ == 0:
                print('[Train] Episode: {:5d}, Loss: {}  '.format(
                    episode+1,
                    self.mean_loss.result()
                ))
                self.mean_loss.reset_states()
                is_best_score = self.metric(self.model.get_weights(), episode)
                if is_best_score:
                    self.model.save_weights(MODEL_FILE)


if __name__ == '__main__':
    worker = Worker()
    worker.run()
