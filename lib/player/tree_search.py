import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras.models import load_model
from itertools import product
from . import Player, InvalidMoveError
from .lib import *
import time

class Node():
    def __init__(self, color):
        self.probs = np.zeros(225)
        self.actions = np.zeros(225)
        self.visits = np.zeros(225)
        self.observed = 0
        self.rewards = np.zeros(225)
        self.color = color
        self.children = [None for i in range(225)]


    def add_child(self, number):
        child = Node(3 - self.color)
        self.children[number] = child
        self.observed = 1


    def is_empty(self):
        return not self.observed


class MCTSPlayer(Player):
    name = 'MCTS'
    def __init__(self, color, timeout=9.9, max_depth=10):
        self.max_depth = max_depth
        self.timeout = timeout
        self.rollout = load_model('./lib/player/supervised_rollout.h5')
        self.policy = load_model('./lib/player/supervised_policy.h5')
        self.color = color
        self.position = np.zeros((15, 15))


    def propagate(self, path, reward):
        current = self.root
        for action in path:
            if current.color == 1:
                current.rewards[action] += reward
            else:
                current.rewards[action] -= reward
            current.visits[action] += 1
            current = current.children[action]
        del path
        return


    def tree_search(self):
        time0 = time.time()
        timeout = min(self.timeout, 60)
        iteration = 1
        while time.time() - time0 < timeout or iteration < 200:
            iteration += 1
            path, reward = self.simulate()
            self.propagate(path, reward)


    def make_rollout(self, position, move):
        last_move = move
        win = 0
        color = 3 - position[last_move[0], last_move[1]]
        actions = np.zeros(225)
        X_white = np.zeros((1, 15, 15, 4))
        X_black = np.zeros((1, 15, 15, 4))
        for line, col in product(range(15), range(15)):
            if position[line, col]:
                X_white[0, line, col, 0] = 1
                X_black[0, line, col, 0] = 1
                if position[line, col] == 1:
                    X_white[0, line, col, 1] = 1
                    X_black[0, line, col, 2] = 1
                else:
                    X_white[0, line, col, 2] = 1
                    X_black[0, line, col, 1] = 1
            else:
                actions[coords_to_num(line, col)] = 1
            X_white[0, line, col, 3] = 1
            X_black[0, line, col, 3] = 1
        if not actions.sum():
            return 0
        win = check_cell(position, last_move[0], last_move[1])
        while not win and self.depth < self.max_depth:
            for line, col in product(range(15), range(15)):
                if position[line, col] == 0:
                    position[line, col] = color
                    if check_cell(position, line, col):
                        return 3 - 2 * color
                    position[line, col] = 0
            if color == 1:
                X = X_white
            else:
                X = X_black
            probs = self.rollout.predict(X
                    , batch_size=1
                    , verbose=0)
            probs *= actions
            probs /= probs.sum()
            action = np.random.choice(225, p=probs[0])
            actions[action] = 0
            if actions.sum() == 0:
                return 0

            last_move = num_to_coords(action)
            position[last_move[0], last_move[1]] = color
            X_white[0, last_move[0], last_move[1], 0] = 1
            X_black[0, last_move[0], last_move[1], 0] = 1
            if color == 1:
                X_white[0, last_move[0], last_move[1], 1] = 1
                X_black[0, last_move[0], last_move[1], 2] = 1
            else:
                X_white[0, last_move[0], last_move[1], 2] = 1
                X_black[0, last_move[0], last_move[1], 1] = 1
            color = 3 - color
            self.depth += 1
            win = check_cell(position, last_move[0], last_move[1])
        if win:
            return 3 - 2 * position[last_move[0], last_move[1]]
        else:
            return 0


    def simulate(self):
        current = self.root
        self.depth = 0
        reward = 0
        new_position = np.copy(self.position)
        path = []
        action = 0
        X = np.zeros((1, 15, 15, 4))
        while self.depth < self.max_depth:
            if current.is_empty():
                current.observed = 1
                X.fill(0)
                for line, col in product(range(15), range(15)):
                    if new_position[line, col]:
                        X[0, line, col, 0] = 1
                        if new_position[line, col] == current.color:
                            X[0, line, col, 1] = 1
                        else:
                            X[0, line, col, 2] = 1
                    else:
                        current.actions[coords_to_num(line, col)] = 1
                    X[0, line, col, 3] = 1
                current.probs = self.policy.predict(X
                        , batch_size=1
                        , verbose=0)
                current.probs *= current.actions
                action = np.argmax(current.probs)
                move = num_to_coords(action)
                new_position[move[0], move[1]] = current.color
                path.append(action)
                if check_cell(new_position, move[0], move[1]):
                    return path, 3 - 2 * current.color
                current.add_child(action)
                reward = self.make_rollout(new_position, move)
                break
            else:
                action = np.argmax((current.rewards > current.visits * 0.85) +\
                        (current.rewards > 0.25 * current.visits) + (current.rewards * current.actions / (1 + current.visits))\
                        + 10 * current.probs * current.actions / (1 + current.visits))
                path.append(action)
                move = num_to_coords(action)
                new_position[move[0], move[1]] = current.color
                if check_cell(new_position, move[0], move[1]):
                    reward = 3 - 2 * current.color
                    return path, reward
                if not current.visits[action]:
                    current.add_child(action)
                current = current.children[action]
                self.depth += 1
        return path, reward


    def _make_move(self, gui):
        self.root = Node(self.color)
        self.position = np.copy(gui.board.board)
        self.tree_search()
        big_values = (self.root.visits > 50) * self.root.rewards\
                / (1 + self.root.visits)
        if np.max(big_values) > 0.8:
            new_action = np.argmax(big_values)
        else:
            new_action = np.argmax(self.root.visits)
        x, y = num_to_coords(new_action)
        del self.root
        gui.board[x, y] = self.color
