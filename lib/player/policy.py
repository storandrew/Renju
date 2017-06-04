import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras.models import load_model
from itertools import product
from . import Player, InvalidMoveError
from .lib import *

class PolicyNetwork(Player):
    name = 'Policy'
    def __init__(self, color):
        self.model = load_model('./lib/player/supervised_policy.h5')
        self.color = color


    def _make_move(self, gui):
        X = np.zeros((1, 15, 15, 4))
        possible = np.zeros(225)
        position = np.copy(gui.board.board)
        win = 0
        for line, col in product(range(15), range(15)):
            if position[line, col] == 0:
                position[line, col] = self.color
                if check_cell(position, line, col):
                    win = 1
                    x, y = line, col
                position[line, col] = 0
        for line, col in product(range(15), range(15)):
            if position[line, col]:
                X[0, line, col, 0] = 1
                if position[line, col] == self.color:
                    X[0, line, col, 1] = 1
                else:
                    X[0, line, col, 2] = 1
            else:
                possible[coords_to_num(line, col)] = 1
            X[0, line, col, 3] = 1
        probs = self.model.predict(X, batch_size=1, verbose=0)
        probs *= possible
        if not win:
            x, y = num_to_coords(np.argmax(probs))
        gui.board[x, y] = self.color
