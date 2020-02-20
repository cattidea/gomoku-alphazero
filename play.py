import os
import threading
import argparse

from actor import ActorCriticModel, Computer, Human
from config import BLACK, HALF, MODEL_FILE, NOTHING, WHITE, height, width
from game import Game
from gui import GUI

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GameManager():

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.game = Game((width, height))

    def play(self, gui=None):
        game = self.game
        flag = True
        while flag:
            for player in (self.player1, self.player2):
                value = game.curr_value
                env = value * game.board
                x, y = player(env)
                if gui is not None:
                    color = 'white' if value == WHITE else 'black'
                    gui.circle(x, y, color=color)
                status = game.play(x, y)
                if status == NOTHING:
                    continue
                if status == BLACK:
                    message = '黑棋胜！'
                elif status == WHITE:
                    message = '白棋胜！'
                elif status == HALF:
                    message = '平局！'
                if gui is not None:
                    gui.messagebox(message)
                    gui.init_board()
                else:
                    print(message)
                flag = False
                break

    def play_with_gui(self):
        gui = GUI()

        def loop():
            while True:
                self.play(gui)
        loop_thread = threading.Thread(target=loop)
        loop_thread.setDaemon(True)
        loop_thread.start()
        gui.tk.mainloop()


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
    parser.add_argument('--prob', action='store_true', help='机器按概率选取操作')
    args = parser.parse_args()
    player1, player2 = get_players(args.mode, args.prob)
    gm = GameManager(player1, player2)
    gm.play_with_gui()
