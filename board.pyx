import numpy as np
cimport numpy as np
cimport cython

cdef enum:
   BLACK = -1
   WHITE = 1
   NOTHING = 2
   TIE = 0

cdef struct Point:
    int x
    int y

cdef class Board():

    cdef public np.ndarray data
    cdef int n_in_row, num_chess
    cdef (int, int) latest_point
    cdef public int width, height, curr_player
    def __init__(self, tuple shape=(8, 8), int n_in_row=5):
        self.data = np.zeros(shape, dtype=np.int8)
        self.n_in_row = n_in_row
        self.width, self.height = shape[0], shape[1]
        self.latest_point = (-1, -1)
        self.num_chess = 0
        self.curr_player = BLACK

    cdef int num_same_chess(self, Point p, int value, Point axis):
        cdef int w = self.data.shape[0]
        cdef int h = self.data.shape[1]
        cdef int cnt = 0, i
        cdef int curr_x, curr_y
        for i in range(1, 5):
            curr_x = p.x + axis.x * i
            curr_y = p.y + axis.y * i
            if curr_x < 0 or curr_x >= w or curr_y < 0 or curr_y >= h:
                break
            if self.data[curr_x, curr_y] != value:
                break
            cnt += 1
        return cnt

    cdef bint check_winner(self, Point p, int value):
        if self.num_same_chess(p, value, Point(-1, 0)) + self.num_same_chess(p, value, Point(1, 0)) >= self.n_in_row-1 or \
            self.num_same_chess(p, value, Point(-1, 1)) + self.num_same_chess(p, value, Point(1, -1)) >= self.n_in_row-1 or \
            self.num_same_chess(p, value, Point(0, -1)) + self.num_same_chess(p, value, Point(0, 1)) >= self.n_in_row-1 or \
            self.num_same_chess(p, value, Point(-1, -1)) + self.num_same_chess(p, value, Point(1, 1)) >= self.n_in_row-1:
            return 1
        return 0

    def new_game(self):
        self.curr_player = BLACK
        self.num_chess = 0
        self.latest_point = (-1, -1)
        self.data[:] = 0

    def move_to(self, x=None, y=None, loc=None):
        if loc is not None:
            x, y = loc // self.height, loc % self.height
        assert x >= 0 and x < self.width and y >= 0 and y < self.height
        assert self.data[x, y] == 0
        self.data[x, y] = self.curr_player
        self.num_chess += 1
        self.latest_point = (x, y)
        self.curr_player = -self.curr_player

    def game_end(self):
        latest_player = -self.curr_player
        if self.latest_point[0] != -1:
            x, y = self.latest_point
            if self.check_winner(Point(x, y), latest_player):
                return True, latest_player
            if self.num_chess == self.width * self.height:
                return True, TIE
        return False, None

    @property
    def availables(self):
        availables = np.where(self.data.reshape((self.width*self.height, )) == 0)[0]
        return availables

    @property
    def state(self):
        curr_state = np.zeros((self.width, self.height, 4), np.int8)
        # 通道一：我方局势
        curr_state[:, :, 0] = self.data == self.curr_player
        # 通道二：敌方局势
        curr_state[:, :, 1] = self.data == -self.curr_player
        # 通道三：上次落子位置
        if self.latest_point is not None:
            curr_state[self.latest_point[0], self.latest_point[1], 2] = 1
        # 通道四：当前玩家是否为先手玩家（是则全为 1，否则全为 0）
        if self.curr_player == BLACK:
            curr_state[:, :, 3] = 1
        return curr_state

