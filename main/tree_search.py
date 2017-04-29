from __future__ import print_function
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from itertools import product


def coords_to_num(x_coord, y_coord):
    return 15 * x_coord + y_coord


def num_to_coords(number):
    return number // 15, number % 15


def coords_to_str(line, col):
    if col >= ord('i') - ord('a'):
        col += 1
    string = chr(col + ord('a'))
    string += str(line)
    return string


def print_board(position):
    print('  |  a|b|c|d|e|f|g|h|j|k|l|m|n|o|p')
    print('__|_______________________________')
    for line, column in product(range(15), range(15)):
        if column == 0:
            print(str(line + 1).zfill(2), end='|  ',)
        endline = '\n' if column == 14 else '|'
        if position[column, line] == 0:
            cell = '_'
        elif position[column, line] == 1:
            cell = 'x'
        else:
            cell = 'o'
        print(cell, end=endline)


def check_position(position):
    for line, column in product(range(15), range(15)):
        if (check_cell(position, line, column)):
            return 1
    return 0


def check_cell(position, line, col):
    for diff in range(5):
        line_in = line - diff > 0 and line + 5 - diff <= 15
        column_in = col - diff > 0 and col + 5 - diff <= 15
        column2_in = col - 5 + diff > 0 and col + diff <= 15
        cell_type = position[line, col]
        if line_in:
            row = position[line-diff : line+5-diff, col]
            if cell_type and not (row - cell_type).any():
                return 1
        if column_in:
            row = position[line, col-diff : col+5-diff]
            if cell_type and not (row - cell_type).any():
                return 1
        if line_in and column_in:
            row = position[line-diff : line+5-diff
                    , col-diff : col+5-diff].diagonal()
            if cell_type and not (row - cell_type).any():
                return 1
        if line_in and column2_in:
            row = np.rot90(position[line-diff : line+5-diff
                , col-4+diff : col+1+diff]).diagonal()
            if cell_type and not (row - cell_type).any():
                return 1
    return 0


class Node():
    def __init__(self, position, color, move=None):
        self.position = position
        self.actions = np.zeros(225)
        self.probs = np.zeros(225)
        self.visits = np.zeros(225)
        self.observed = np.zeros(225)
        self.rewards = np.zeros(225)
        self.number = None if move == None else\
                coords_to_num(move[0], move[1])
        for line, col in product(range(15), range(15)):
            if not position[line, col]:
                self.actions[coords_to_num(line, col)] = 1
        self.color = color
        self.children = []
        self.threshold = threshold


    def add_child(self, move):
        new_position = np.copy(self.position)
        new_position[move[0], move[1]] = self.color
        child = Node(new_position, 3 - self.color, move)
        self.children.append(child)
        number = coords_to_num(move[0], move[1])
        self.visits[number] = 1
        self.observed[number] = 1


    def is_explored(self):
        obs = len(self.children)
        return obs == self.threshold or obs == self.actions.sum()


    def is_empty(self):
        return not len(self.children)


class MCTSPlayer():
    def __init__(self, mode):
        self.rollout = load_model('supervised_rollout.h5')
        self.policy = load_model('supervised_policy2.h5')
        self.value = load_model('supervised_value2.h5')
        self.mode = 1 if mode == 'black' else 2


    def __del__(self):
        del self.policy
        del self.value
        del self.rollout


    def propagate(self, path, reward):
        current = self.root
        for node in path:
            number = node.number
            current.rewards[number] += reward
            current.visits[number] += 1
            current = node
        del path
        return


    def tree_search(self, iterations):
        for i in range(iterations):
            print('\rSimulation', i, end='')
            path, reward = self.simulate()
            self.propagate(path, reward)


    def make_rollout(self, position, move):
        last_move = move
        color = 3 - position[last_move[0], last_move[1]]
        while not check_cell(position, last_move[0], last_move[1]):
            actions = np.zeros(225)
            X = np.zeros((1, 15, 15, 4))
            for line, col in product(range(15), range(15)):
                if position[line, col]:
                    X[0, line, col, 0] = 1
                    if position[line, col] == color:
                        X[0, line, col, 1] = 1
                    else:
                        X[0, line, col, 2] = 1
                else:
                    actions[coords_to_num(line, col)] = 1
                X[0, line, col, 3] = 1
            if not actions.sum():
                return 0
            probs = self.rollout.predict_proba(X
                    , batch_size=1
                    , verbose=0)
            action = np.argmax(probs * actions)

            last_move = num_to_coords(action)
            position[last_move[0], last_move[1]] = color
            color = 3 - color
        return 1 if position[last_move[0], last_move[1]] == 1 else -1


    def simulate(self):
        current = self.root
        path = []
        while 1:
            if current.is_empty():
                X = np.zeros((1, 15, 15, 4))
                for line, col in product(range(15), range(15)):
                    if current.position[line, col]:
                        X[0, line, col, 0] = 1
                        if current.position[line, col]\
                                == current.color:
                            X[0, line, col, 1] = 1
                        else:
                            X[0, line, col, 2] = 1
                    X[0, line, col, 3] = 1
                current.probs = self.policy.predict_proba(X
                        , batch_size=1
                        , verbose=0)
                action = np.argmax(current.probs * current.actions)
                move = num_to_coords(action)
                current.add_child(move)
                position = np.copy(current.children[-1].position)
                reward = self.make_rollout(position, move)
                path.append(current.children[-1])
                break
            if current.is_explored():
                print_board(current.position)
                action = np.argmax(current.probs * current.actions / (1 + current.visits))
                move = num_to_coords(action)
                print(coords_to_str(move[0], move[1]))
            else:
                action = np.argmax(current.probs * current.actions\
                        * (1 - current.observed))
                move = num_to_coords(action)
                current.add_child(move)
            for child in current.children:
                if child.number == action:
                    print('hui')
                    current = child
            path.append(current)
        return path, reward


    def move(self, position):
        self.root = Node(position, self.mode)
        self.tree_search(100)
        new_action = np.argmax(self.root.visits)
        x, y = num_to_coords(new_action)
        del self.root
        return x, y


class Player():
    def __init__(self):
        pass


    def move(self, position):
        pass


class HumanPlayer(Player):
    def __init__(self, mode):
        pass
    def move(self, position):
        print_board(position)
        letter = 'abcdefghjklmnop'
        digit = '0123456789'
        while 1:
            move = input()
            if len(move) < 2 or len(move) > 3:
                print('Invalid input')
                continue
            if move[0] not in letter:
                print('Invalid input')
                continue
            if move[1] not in digit:
                print('Invalid input')
                continue
            if len(move) == 3 and move[2] not in digit:
                print('Invalid input')
                continue
            x, y = ord(move[0]) - ord('a'), int(move[1:]) - 1
            if ord(move[0]) > ord('i'):
                x -= 1
            if y > 14 or y < 0:
                print('Invalid input')
                continue
            if position[x, y]:
                print('Invalid input')
                continue
            return x, y


class RandomPlayer(Player):
    def move(self, position):
        moves = []
        for line, col in product(range(15), range(15)):
            if not position[line, col]:
                moves.append(coords_to_num(line, col))
        return num_to_coords(np.random.choice(moves))


class PolicyNetwork(Player):
    def __init__(self, mode):
        self.model = load_model('supervised_policy2.h5')
        self.mode = 1 if mode == 'black' else 2


    def move(self, position):
        X = np.zeros((1, 15, 15, 4))
        possible = np.zeros(225)
        for line, col in product(range(15), range(15)):
            if position[line, col]:
                X[0, line, col, 0] = 1
                if position[line, col] == self.mode:
                    X[0, line, col, 1] = 1
                else:
                    X[0, line, col, 2] = 1
            else:
                possible[coords_to_num(line, col)] = 1
            X[0, line, col, 3] = 1
        probs = self.model.predict(X, batch_size=1, verbose=0)
        probs *= possible
        best_move = np.argmax(probs)
        return num_to_coords(best_move)


class Referee():
    def __init__(self, crosses, noughts):
        self.crosses = crosses
        self.noughts = noughts


    def play(self):
        position = np.zeros((15, 15))
        while 1:
            line, col = self.crosses.move(position)
            position[line, col] = 1
            if check_cell(position, line, col):
                print('BLACK WIN!\n')
                print_board(position)
                return 1
            line, col = self.noughts.move(position)
            position[line, col] = 2
            if check_cell(position, line, col):
                print('WHITE WIN!\n')
                print_board(position)
                return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('black')
    parser.add_argument('white')
    args = parser.parse_args()
    if args.black == 'me':
        player1 = HumanPlayer('black')
    elif args.black == 'policy':
        player1 = PolicyNetwork('black')
    elif args.black == 'mcts':
        player1 = MCTSPlayer('black')

    if args.white == 'me':
        player2 = HumanPlayer('white')
    elif args.white == 'policy':
        player2 = PolicyNetwork('white')
    elif args.white == 'mcts':
        player2 = MCTSPlayer('white')

    ref = Referee(player1, player2)
    ref.play()


if __name__ == "__main__":
    main()
