import sys
import os
sys.path.append(os.getcwd()) 

import numpy as np
import util
import logging
import renju
import backend
import time

from keras.models import load_model
from itertools import product

def coords_to_str(line, col):
    if line >= ord('i') - ord('a'):
        line += 1
    string = chr(line + ord('a'))
    string += str(col + 1)
    return string


def coords_to_num(x_coord, y_coord):
    return 15 * x_coord + y_coord


def num_to_coords(number):
    return number // 15, number % 15


def check_cell(position, line, col):
    cur = position[line, col]
    i, j = line - 1, line + 1
    while i >= 0 and position[i, col] == cur:
        i -= 1
    while j < 15 and position[j, col] == cur:
        j += 1
    if j - i - 1 >= 5:
        return 1
    i, j = col - 1, col + 1
    while i >= 0 and position[line, i] == cur:
        i -= 1
    while j < 15 and position[line, j] == cur:
        j += 1
    if j - i - 1 >= 5:
        return 1
    i, j = -1, 1
    while line + i >= 0 and col + i >= 0 and position[line + i, col + i] == cur:
        i -= 1
    while line + j < 15 and col + j < 15 and position[line + j, col + j] == cur:
        j += 1
    if j - i - 1 >= 5:
        return 1
    i, j = -1, 1
    while line + i >= 0 and col - i < 15 and position[line + i, col - i] == cur:
        i -= 1
    while line + j < 15 and col - j >= 0 and position[line + j, col - j] == cur:
        j += 1
    if j - i - 1 >= 5:
        return 1
    return 0


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


class MCTS():
    def __init__(self, color=1, timeout=9.9, max_depth=10):
        self.max_depth = max_depth
        self.timeout = timeout
        self.rollout = load_model('./supervised_rollout.h5')
        self.policy = load_model('./supervised_policy.h5')
        self.color = color
        self.position = np.zeros((15, 15))


    def __del__(self):
        del self.policy
        del self.rollout


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
        while time.time() - time0 < timeout:
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
                        (current.rewards > 0.25 * current.visits) +\
                        (current.rewards * current.actions / (1 + current.visits))\
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


    def make_move(self, position, color):
        self.color = color
        self.root = Node(self.color)
        self.position = position
        self.tree_search()
        for line, col in product(range(15), range(15)):
                if position[line, col] == 0:
                    position[line, col] = color
                    if check_cell(position, line, col):
                        return coords_to_str(col, line)
                    position[line, col] = 0
        big_values = (self.root.visits > 50) * self.root.rewards\
                / (1 + self.root.visits)
        if np.max(big_values) > 0.5:
            new_action = np.argmax(big_values)
        else:
            new_action = np.argmax(self.root.visits)
        x, y = num_to_coords(new_action)
        return coords_to_str(y, x)


def main():
    logging.basicConfig(filename='andrew.log', level=logging.DEBUG)
    logging.debug("Start Andrew's player backend...")
    mcts = MCTS()

    try:
        while True:
            logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()
            logging.debug('Board:\n' + str(game.board()))
            position = game.board()
            color = 1 if np.sum(position) == 0 else 2
            for line, col in product(range(15), range(15)):
                position[line, col] *= -1
                if position[line, col] == -1:
                    position[line, col] = 2

            move = mcts.make_move(position, color)
            logging.debug('choose move: ' + move)
            backend.move(move)
            logging.debug('make move: ' + move)
    except:
        logging.debug('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
