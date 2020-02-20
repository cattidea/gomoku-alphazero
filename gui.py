import threading
from tkinter import *
from tkinter import messagebox

import numpy as np

from actor import POINT_QUEUE
from config import *
from game import Game

CW = 30
R = 10


class Scaler():
    """ 坐标变换器
    方便对坐标进行放缩
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def bind(self, start, end):
        self.new_start = start
        self.new_end = end
        return self

    def inverse(self):
        return Scaler(self.new_start, self.new_end).bind(self.start, self.end)

    def __call__(self, value):
        length = self.end - self.start
        new_length = self.new_end - self.new_start
        new_value = (value - self.start) / length * new_length + self.new_start
        return new_value


class BinaryScaler():
    """ 二元坐标变换器 """
    def __init__(self, start_x, start_y, end_x, end_y):
        self.X = Scaler(start_x, end_x)
        self.Y = Scaler(start_y, end_y)

    def bind(self, start_x, start_y, end_x, end_y):
        self.X.bind(start_x, end_x)
        self.Y.bind(start_y, end_y)
        return self

    def inverse(self):
        bs = BinaryScaler(self.X.new_start, self.Y.new_start,
                          self.X.new_end, self.Y.new_end)
        bs.bind(self.X.start, self.Y.start, self.X.end, self.Y.end)
        return bs

    def __call__(self, value_x, value_y):
        return self.X(value_x), self.Y(value_y)


class GUI():
    """ 棋盘 UI """
    def __init__(self):
        self.tk = Tk()
        self.tk.geometry("{}x{}".format(width*CW+100, height*CW+100))
        self.bc = None
        self.canvas = None
        self.init_canvas()
        self.status = NOTHING

    def init_canvas(self):
        canvas_width, canvas_height = width*CW, height*CW
        bc = BinaryScaler(0, 0, canvas_width,
                          canvas_height).bind(-1, height, width, -1)
        bc_ = bc.inverse()
        self.bc = bc
        self.canvas_width, self.canvas_height = canvas_width, canvas_height

        canvas = Canvas(self.tk, width=canvas_width,
                        height=canvas_height, bg='orange')
        self.canvas = canvas
        self.init_board()

        def on_click(event):
            x, y = bc(event.x, event.y)
            x, y = int(x + 0.5), int(y + 0.5)
            POINT_QUEUE.put((x, y))

        canvas.bind("<ButtonRelease-1>", on_click)

    def line(self, x1, y1, x2, y2):
        bc_ = self.bc.inverse()
        x1, y1 = bc_(x1, y1)
        x2, y2 = bc_(x2, y2)
        self.canvas.create_line(x1, y1, x2, y2)

    def init_board(self):
        bc_ = self.bc.inverse()
        canvas = self.canvas
        canvas.create_rectangle(0, 0, self.canvas_width,
                                self.canvas_height, fill='orange')
        for i in range(height):
            self.line(0, i, width-1, i)
        for i in range(width):
            self.line(i, 0, i, height-1)
        canvas.place(x=50, y=50, anchor='nw')

    def circle(self, x, y, radius=R, color='blue'):
        bc_ = self.bc.inverse()
        x_pix, y_pix = bc_(x, y)
        self.canvas.create_oval(x_pix-radius, y_pix -
                                radius, x_pix+radius, y_pix+radius, fill=color)

    def messagebox(self, message):
        messagebox.showinfo('INFO', message)
