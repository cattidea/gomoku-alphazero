import os
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

from config import *
from board import Board
from policy import ActorCriticModel, mean_policy_value_fn
from ui import GUI, TerminalUI, HeadlessUI
from mcts import MCTS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Memory():
    def __init__(self):
        self.states = []
        self.mcts_probs = []
        self.cnt = 0

    def store(self, state, mcts_probs):
        self.states.append(state)
        self.mcts_probs.append(mcts_probs)
        self.cnt += 1

    def clear(self):
        self.states.clear()
        self.mcts_probs.clear()
        self.cnt = 0


class DataBuffer(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.cache = {
            BLACK: Memory(),
            WHITE: Memory()
        }

    def clear_cache(self):
        self.cache[BLACK].clear()
        self.cache[WHITE].clear()

    def collect(self, player, state, mcts_prob):
        self.cache[player].store(state, mcts_prob)

    def get_discounted_rewards(self, player, winner):
        if winner == TIE:
            return [0 for _ in range(self.cache[player].cnt)]
        reward_sum = 0.
        discounted_rewards = []
        rewards = [0 for _ in range(self.cache[player].cnt)]
        rewards[0] = player * winner
        for reward in rewards:
            reward_sum = reward + REWARD_GAMMA * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        return discounted_rewards

    def end_episode(self, winner):

        discounted_rewards = {
            BLACK: self.get_discounted_rewards(BLACK, winner),
            WHITE: self.get_discounted_rewards(WHITE, winner)
        }

        play_data = list(zip(self.cache[BLACK].states, self.cache[BLACK].mcts_probs, discounted_rewards[BLACK])) + \
                    list(zip(self.cache[WHITE].states, self.cache[WHITE].mcts_probs, discounted_rewards[WHITE]))

        play_data = self.augment_and_append(play_data)
        self.clear_cache()

    def augment_and_append(self, play_data):
        for state, mcts_prob, reward in play_data:
            if mcts_prob is None:
                continue
            for num_revo in range(4):
                for is_flip in (True, False):
                    state_aug = np.rot90(state, num_revo)
                    mcts_prob_aug = mcts_prob.reshape(width, height)
                    mcts_prob_aug = np.rot90(mcts_prob_aug, num_revo)
                    if is_flip:
                        state_aug = np.flip(state_aug)
                        mcts_prob_aug = np.flip(mcts_prob_aug)
                    mcts_prob_aug = mcts_prob_aug.flatten()
                    self.append((state_aug, mcts_prob_aug, reward))


class Player():

    def __init__(self):
        self.ui = None

    def bind(self, color, data_buffer, ui):
        self.color = color
        self.data_buffer = data_buffer
        self.ui = ui

    def __call__(self, board, **kwargs):
        raise NotImplementedError

    @staticmethod
    def move_to_location(loc):
        x, y = loc // width, loc % width
        return x, y


class Computer(Player):

    def __init__(self, weights=None, prob=False):
        super().__init__()
        self.model = ActorCriticModel()
        self.prob = prob
        if weights is not None:
            self.model.build(input_shape=(None, width, height, 1))
            self.model.load_weights(weights)

    def __call__(self, board, **kwargs):
        curr_state = np.expand_dims(board.state, axis=0)

        logits, _ = self.model(tf.convert_to_tensor(
            curr_state, dtype=tf.float32))
        mask = tf.reshape(curr_state == 0, (width*height, ))
        logits = tf.where(mask, logits, -np.inf)
        probs = tf.nn.softmax(logits)

        action = np.random.choice(
            height*width, p=probs.numpy()[0]) if self.prob else np.argmax(probs)
        x, y = self.move_to_location(action)
        return x, y


class MCTSPlayer(Player):

    def __init__(self, c_puct=5, n_playout=2000):
        super().__init__()
        self.mcts = MCTS(mean_policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __call__(self, board, **kwargs):
        sensible_moves = board.availables
        assert len(sensible_moves) > 0
        move = self.mcts.get_move(board)
        self.mcts.update_with_move(-1)
        x, y = self.move_to_location(move)
        return x, y, None


class MCTSA3CPlayer(Player):
    def __init__(self, weights=None, c_puct=5, n_playout=2000):
        self.model = ActorCriticModel()
        if weights is not None:
            self.model.build(input_shape=(None, width, height, 4))
            self.model.load_weights(weights)
        self.mcts = MCTS(self.model.policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __call__(self, board, is_selfplay=False, temp=1e-3):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width*board.height)
        assert len(sensible_moves) > 0
        state = board.state
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25 *
                np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        x, y = self.move_to_location(move)
        return x, y, move_probs


class Human(Player):

    def __init__(self):
        super().__init__()

    def __call__(self, board, **kwargs):
        while True:
            x, y = self.ui.input()
            if x >= 0 and x < width and y >= 0 and y < height and \
                    board.data[x, y] == 0:
                return x, y, None


class Game():

    def __init__(self, player1, player2, ui=None):
        self.player1 = player1
        self.player2 = player2
        self.ui = ui
        self.board = Board((width, height))
        self.data_buffer = DataBuffer(maxlen=BUFFER_LENGTH)
        self.player1.bind(BLACK, self.data_buffer, ui)
        self.player2.bind(WHITE, self.data_buffer, ui)

    def play(self, is_selfplay=False):
        if is_selfplay:
            assert self.player1 is self.player2
        board = self.board
        board.new_game()
        self.ui.reset()
        while True:
            for player in (self.player1, self.player2):
                x, y, move_probs = player(board, is_selfplay=is_selfplay)
                if is_selfplay:
                    self.data_buffer.collect(
                        board.curr_player, board.state, move_probs)
                board.move_to(x, y)
                is_end, winner = board.game_end()
                self.ui.render(board.data, last_move=(x, y))
                if not is_end:
                    continue
                message = {
                    BLACK: '黑棋胜！',
                    WHITE: '白棋胜！',
                    TIE: '平局！'
                }[winner]
                self.ui.message(message)
                if is_selfplay:
                    self.data_buffer.end_episode(winner)
                return winner

    def start(self, is_selfplay=False):
        def loop():
            while True:
                self.play(is_selfplay=is_selfplay)
        self.ui.game_start(loop)


def get_players(mode_str, prob):
    weights = MODEL_FILE
    modes = mode_str.lower().split('v')
    players = []
    for mode in modes:
        players.append(Human() if mode[0] == 'p' else Computer(weights, prob))
    return players


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gomoku A3C')
    parser.add_argument('--mode', default='pve', choices=[
                        'pvp', 'pve', 'evp', 'eve'], help='恢复模型继续训练')
    parser.add_argument('--ui', default='gui', choices=[
                        'gui', 'terminal', 'no'], help='UI 类型')
    parser.add_argument('--prob', action='store_true', help='机器按概率选取操作')
    args = parser.parse_args()
    ui = {'gui': GUI, 'terminal': TerminalUI, 'no': HeadlessUI}[args.ui]()
    player1, player2 = get_players(args.mode, args.prob)
    player1, player2 = Human(), MCTSPlayer(c_puct=5, n_playout=1000)
    # player1, player2 = MCTSA3CPlayer(c_puct=5, n_playout=10), MCTSA3CPlayer(c_puct=5, n_playout=10)
    # player1 = player2 = MCTSA3CPlayer(c_puct=5, n_playout=10)
    # player1, player2 = MCTSPlayer(c_puct=5, n_playout=10), MCTSPlayer(c_puct=5, n_playout=10)
    game = Game(player1, player2, ui)
    game.start(is_selfplay=False)
