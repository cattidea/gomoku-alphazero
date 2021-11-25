import argparse
import gc
import os
import random
import time

import numpy as np
import tensorflow as tf

from config import *
from play import Game, MCTSAlphaZeroPlayer, MCTSPlayer
from policy import AlphaZeroError
from ui import HeadlessUI


class DataAugmentor:
    """数据增强器
    对原数据进行旋转 + 对称，共八种增强方式"""

    def __init__(self, board_shape, rotate=True, flip=True):
        self.board_shape = board_shape
        self.rotate = rotate and board_shape[0] == board_shape[1]
        self.flip = flip

    def __call__(self, data_batch):
        data_batch_aug = []
        for state, mcts_prob, reward in data_batch:
            state_aug = state
            mcts_prob_aug = mcts_prob.reshape(self.board_shape)
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


class AlphaZeroMetric:
    """AlphaZero 性能评估器"""

    def __init__(self, board_shape, n_playout=400):
        self.n_playout = n_playout
        self.n_playout_mcts = 1000
        self.best_score = -np.inf
        self.board_shape = board_shape

    def __call__(self, weights, episode=0, n_games=10):
        assert n_games % 2 == 0

        mcts_alphazero_player = MCTSAlphaZeroPlayer(c_puct=5, n_playout=self.n_playout, board_shape=self.board_shape)
        mcts_alphazero_player.model.build(input_shape=(None, *self.board_shape, CHANNELS))
        mcts_alphazero_player.model.set_weights(weights)
        mcts_player = MCTSPlayer(c_puct=5, n_playout=self.n_playout_mcts, board_shape=self.board_shape)
        game = Game(mcts_alphazero_player, mcts_player, board_shape=self.board_shape, ui=HeadlessUI(self.board_shape))
        scores = {WIN: 0, LOSE: 0, TIE: 0}
        score = 0.0
        for idx in range(n_games):
            winner = game.play(is_selfplay=False)
            res = winner * mcts_alphazero_player.color
            scores[res] += 1
            game.switch_players()
            print("[Testing] Episode: {:5d}, Game: {:2d}, Score: {:2d} ".format(episode + 1, idx, res), end="\r")
        for key in scores:
            score += key * scores[key]
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(
            "[Test]  Episode: {:5d}, MCTS n_playout: {:6d}, Win: {:2d}, Lose: {:2d}, Tie: {:2d}, Score: {:.2f} @{} ".format(
                episode + 1, self.n_playout_mcts, scores[WIN], scores[LOSE], scores[TIE], score, now
            )
        )
        if score > self.best_score:
            self.best_score = score
            if score == n_games:
                self.best_score = -np.inf
                self.n_playout_mcts += 500
            return True
        return False


class Worker:
    def __init__(
        self,
        board_shape,
        resume=False,
        weights_file=MODEL_FILE,
        buffer_file=BUFFER_FILE,
        lr=LEARNING_RATE,
        freeze_cnn=False,
    ):
        self.weights_file = weights_file
        self.buffer_file = buffer_file
        self.board_shape = board_shape
        self.width, self.height = board_shape
        self.player = MCTSAlphaZeroPlayer(c_puct=5, n_playout=400, board_shape=board_shape)
        self.model = self.player.model
        self.model.build(input_shape=(None, *board_shape, CHANNELS))
        self.opt = tf.keras.optimizers.Adam(lr)
        self.loss_object = AlphaZeroError()
        self.mean_loss = tf.keras.metrics.Mean(name="train_loss")
        self.game = Game(self.player, self.player, board_shape=board_shape, ui=HeadlessUI(board_shape))
        self.data_aug = DataAugmentor(board_shape, rotate=True, flip=True)
        self.metric = AlphaZeroMetric(board_shape=board_shape, n_playout=400)

        if resume:
            self.model.load_weights(self.weights_file)
            print("Loaded model successfully.")
            if os.path.exists(self.buffer_file):
                self.game.data_buffer.load(self.buffer_file)
                print("Loaded buffer ({} items) successfully.".format(len(self.game.data_buffer)))

        if freeze_cnn:
            for layer in self.model.cnn_layers:
                layer.trainable = False

        self.model.summary()

    def run(self):
        for episode in range(MAX_EPISODE):
            winner = self.game.play(is_selfplay=True)
            gc.collect()

            total_loss = tf.constant(0)

            for epoch in range(EPOCHS):
                mini_batch = random.sample(self.game.data_buffer, min(BATCH_SIZE, len(self.game.data_buffer) // 2))
                mini_batch = self.data_aug(mini_batch)
                states_batch, mcts_probs_batch, rewards_batch = zip(*mini_batch)
                states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
                mcts_probs_batch = tf.convert_to_tensor(mcts_probs_batch, dtype=tf.float32)
                rewards_batch = tf.convert_to_tensor(np.expand_dims(rewards_batch, axis=-1), dtype=tf.float32)

                with tf.GradientTape() as tape:
                    policy, values = self.model(states_batch, training=True)

                    total_loss = self.loss_object(
                        mcts_probs=mcts_probs_batch, policy=policy, rewards=rewards_batch, values=values
                    )

                grads = tape.gradient(total_loss, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

                self.mean_loss(total_loss)

                print(
                    "[Training] Episode: {:5d}, Epoch: {:2d}, Winner: {:5s}, Loss: {}  ".format(
                        episode + 1, epoch + 1, COLOR[winner], total_loss.numpy()
                    ),
                    end="\r",
                )

            if (episode + 1) % CHECK_FREQ == 0:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(
                    "[Train] Episode: {:5d}, Loss: {} @{}                        ".format(
                        episode + 1, self.mean_loss.result(), now
                    ),
                )
                self.mean_loss.reset_states()
                is_best_score = self.metric(self.model.get_weights(), episode)
                if is_best_score:
                    self.model.save_weights(f"data/model-{self.width}x{self.height}#{N_IN_ROW}.h5")
                    self.game.data_buffer.save(f"data/buffer-{self.width}x{self.height}#{N_IN_ROW}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku AlphaZero")
    parser.add_argument("--resume", action="store_true", help="恢复模型继续训练")
    parser.add_argument("--weights", default=MODEL_FILE, help="预训练权重存储位置")
    parser.add_argument("--buffer", default=BUFFER_FILE, help="经验池存储位置")
    parser.add_argument("--lr", default=LEARNING_RATE, type=float, help="训练时学习率")
    parser.add_argument("--width", default=WIDTH, type=int, help="棋盘水平宽度")
    parser.add_argument("--height", default=HEIGHT, type=int, help="棋盘竖直宽度")
    parser.add_argument("--freeze-cnn", action="store_true", help="训练时冻结 CNN 部分")
    args = parser.parse_args()

    worker = Worker(
        board_shape=(args.width, args.height),
        resume=args.resume,
        weights_file=args.weights,
        buffer_file=args.buffer,
        lr=args.lr,
        freeze_cnn=args.freeze_cnn,
    )
    worker.run()
