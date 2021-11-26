import argparse
from collections import deque
from functools import partial

import h5py
import numpy as np

from board import Board
from config import *
from mcts import MCTS
from policy import PolicyValueModelResNet as PolicyValueModel
from policy import mean_policy_value_fn
from ui import GUI, HeadlessUI, TerminalUI


class Memory:
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
        self.cache = {BLACK: Memory(), WHITE: Memory()}

    def clear_cache(self):
        self.cache[BLACK].clear()
        self.cache[WHITE].clear()

    def collect(self, player, state, mcts_prob):
        self.cache[player].store(state, mcts_prob)

    def get_discounted_rewards(self, player, winner):
        if winner == TIE:
            return [0 for _ in range(self.cache[player].cnt)]
        reward_sum = 0.0
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
            WHITE: self.get_discounted_rewards(WHITE, winner),
        }

        play_data = list(zip(self.cache[BLACK].states, self.cache[BLACK].mcts_probs, discounted_rewards[BLACK])) + list(
            zip(self.cache[WHITE].states, self.cache[WHITE].mcts_probs, discounted_rewards[WHITE])
        )

        self.extend(play_data)
        self.clear_cache()

    def save(self, filename):
        states, mcts_probs, rewards = zip(*self)
        f = h5py.File(filename, "w")
        f["states"] = states
        f["mcts_probs"] = mcts_probs
        f["rewards"] = rewards
        f.close()

    def load(self, filename):
        f = h5py.File(filename, "r")
        states = f["states"]
        mcts_probs = f["mcts_probs"]
        rewards = f["rewards"]
        self.extend(zip(states, mcts_probs, rewards))
        f.close()


class Player:
    """玩家基类"""

    def __init__(self, board_shape):
        self.ui = None
        self.width, self.height = board_shape

    def bind(self, color, data_buffer, ui):
        self.color = color
        self.data_buffer = data_buffer
        self.ui = ui

    def __call__(self, board, **kwargs):
        raise NotImplementedError

    def reset_player(self):
        pass

    def move_to_location(self, loc):
        x, y = loc // self.width, loc % self.height
        return x, y


class MCTSPlayer(Player):
    """纯 MCTS 玩家"""

    def __init__(self, board_shape, c_puct=5, n_playout=2000):
        super().__init__(board_shape)
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


class MCTSAlphaZeroPlayer(Player):
    """AlphaZero 玩家"""

    def __init__(self, board_shape, weights=None, c_puct=5, n_playout=2000):
        super().__init__(board_shape)
        self.model = PolicyValueModel(*board_shape)
        if weights is not None:
            self.model.build(input_shape=(None, *board_shape, CHANNELS))
            self.model.load_weights(weights)
        self.mcts = MCTS(self.model.policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __call__(self, board, is_selfplay=False, temp=1e-3):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        assert len(sensible_moves) > 0
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if is_selfplay:
            move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        x, y = self.move_to_location(move)
        return x, y, move_probs


class Human(Player):
    """人类玩家"""

    def __init__(self, board_shape):
        super().__init__(board_shape)

    def __call__(self, board, **kwargs):
        while True:
            x, y = self.ui.input()
            if x >= 0 and x < self.width and y >= 0 and y < self.height and board.data[x, y] == 0:
                return x, y, None


class Game:
    def __init__(self, player1, player2, board_shape, ui=None):
        self.player1 = player1
        self.player2 = player2
        self.ui = ui
        self.board = Board(board_shape, n_in_row=N_IN_ROW)
        self.data_buffer = DataBuffer(maxlen=BUFFER_LENGTH)
        self.player1.bind(BLACK, self.data_buffer, ui)
        self.player2.bind(WHITE, self.data_buffer, ui)

    def switch_players(self):
        self.player1, self.player2 = self.player2, self.player1
        self.player1.color, self.player2.color = self.player2.color, self.player1.color

    def play(self, is_selfplay=False, reverse=False):
        if is_selfplay:
            assert self.player1 is self.player2
        board = self.board
        board.new_game()
        self.ui.reset()
        self.player1.reset_player()
        self.player2.reset_player()
        while True:
            for player in (self.player1, self.player2):
                temp = 1.0 if is_selfplay else 1e-3
                x, y, move_probs = player(board, is_selfplay=is_selfplay, temp=temp)
                if is_selfplay:
                    self.data_buffer.collect(board.curr_player, board.state, move_probs)
                board.move_to(x, y)
                is_end, winner = board.game_end()
                self.ui.render(board.data, last_move=(x, y))
                if not is_end:
                    continue
                message = {BLACK: "黑棋胜！", WHITE: "白棋胜！", TIE: "平局！"}[winner]
                self.ui.message(message)
                if is_selfplay:
                    self.data_buffer.end_episode(winner)
                return winner

    def start(self, is_selfplay=False):
        def loop():
            while True:
                self.play(is_selfplay=is_selfplay)

        self.ui.game_start(loop)


def get_players(
    mode_str,
    width,
    height,
    weights=MODEL_FILE,
    ai_type="alphazero",
    ai_n_playout=2000,
    is_selfplay=False,
):
    AiType = partial(MCTSAlphaZeroPlayer, weights=weights) if ai_type == "alphazero" else MCTSPlayer
    if is_selfplay:
        assert mode_str == "eve", "只有 eve 模式才可以进行自我对局"
        player = AiType(n_playout=ai_n_playout, board_shape=(width, height))
        return (player, player)

    modes = mode_str.lower().split("v")
    players = []
    for mode in modes:
        players.append(
            Human(board_shape=(width, height))
            if mode[0] == "p"
            else AiType(n_playout=ai_n_playout, board_shape=(width, height))
        )
    return players


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku AlphaZero")
    parser.add_argument("--mode", default="pve", choices=["pvp", "pve", "evp", "eve"], help="对战模式（pvp/pve/evp/eve）")
    parser.add_argument("--ui", default="gui", choices=["gui", "terminal", "no"], help="UI 类型")
    parser.add_argument("--weights", default=MODEL_FILE, help="预训练权重存储位置")
    parser.add_argument("--width", default=WIDTH, type=int, help="棋盘水平宽度")
    parser.add_argument("--height", default=HEIGHT, type=int, help="棋盘竖直宽度")
    parser.add_argument(
        "--ai-type",
        default="alphazero",
        choices=["alphazero", "pure-mcts"],
        help="AI 类型（alpha-zero/pure-mcts）",
    )
    parser.add_argument("--ai-n-playout", default=2000, type=int, help="AI MCTS 推演步数")
    parser.add_argument("--selfplay", action="store_true", help="开启 SelfPlay 模式")
    args = parser.parse_args()

    ui = {"gui": GUI, "terminal": TerminalUI, "no": HeadlessUI}[args.ui](board_shape=(args.width, args.height))
    player1, player2 = get_players(
        mode_str=args.mode,
        width=args.width,
        height=args.height,
        weights=args.weights,
        ai_type=args.ai_type,
        ai_n_playout=args.ai_n_playout,
        is_selfplay=args.selfplay,
    )
    game = Game(
        player1,
        player2,
        board_shape=(args.width, args.height),
        ui=ui,
    )
    game.start(is_selfplay=args.selfplay)
