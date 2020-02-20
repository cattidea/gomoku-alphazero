import numpy as np
cimport numpy as np
cimport cython

cdef enum:
   BLACK = -1
   WHITE = 1
   NOTHING = 2
   HALF = 0

cdef struct Point:
    int x
    int y

cdef np.ndarray[np.int8_t, ndim=2] init_board(shape=(8, 8)):
    return np.zeros(shape, dtype=np.int8)

cdef void move(np.ndarray[np.int8_t, ndim=2] board, Point p, int value):
    board[p.x, p.y] = value

cdef bint check_winner(np.ndarray[np.int8_t, ndim=2] board, Point p, int value):
    if num_same_chess(board, p, value, Point(-1, 0)) + num_same_chess(board, p, value, Point(1, 0)) >= 4 or \
        num_same_chess(board, p, value, Point(-1, 1)) + num_same_chess(board, p, value, Point(1, -1)) >= 4 or \
        num_same_chess(board, p, value, Point(0, -1)) + num_same_chess(board, p, value, Point(0, 1)) >= 4 or \
        num_same_chess(board, p, value, Point(-1, -1)) + num_same_chess(board, p, value, Point(1, 1)) >= 4:
        return 1
    return 0

cdef int num_same_chess(np.ndarray[np.int8_t, ndim=2] board, Point p, int value, Point axis):
    cdef int w = board.shape[0]
    cdef int h = board.shape[1]
    cdef int cnt = 0
    cdef int curr_x, curr_y
    for i in range(1, 5):
        curr_x = p.x + axis.x * i
        curr_y = p.y + axis.y * i
        if curr_x < 0 or curr_x >= w or curr_y < 0 or curr_y >= h:
            break
        if board[curr_x, curr_y] != value:
            break
        cnt += 1
    return cnt

cdef void clear_board(np.ndarray[np.int8_t, ndim=2] board):
    board[:] = 0

class Game():

    def __init__(self, shape=(8, 8)):
        cdef np.ndarray[np.int8_t, ndim=2] board = init_board(shape=shape)
        self.width, self.height = shape
        self.board = board
        self.num_chess = 0
        self.curr_value = BLACK
        self.score = {BLACK: 0, WHITE: 0}

    def new_game(self):
        self.curr_value = BLACK
        self.num_chess = 0
        clear_board(self.board)

    def move(self, x, y, value):
        assert x < self.width and y < self.height
        assert self.board[x, y] == 0
        move(self.board, Point(x, y), value)
        self.num_chess += 1
        if check_winner(self.board, Point(x, y), value):
            return value
        if self.num_chess == self.width * self.height:
            return HALF
        return NOTHING

    def play(self, x, y):
        status = self.move(x, y, self.curr_value)
        if status == BLACK or status == WHITE:
            self.score[status] += 1
            self.score[-status] -= 1
            self.new_game()
        elif status == HALF:
            self.new_game()
        else:
            self.curr_value = -self.curr_value
        return status
