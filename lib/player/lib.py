from ..board import black, white, empty, Board, InvalidMoveError
import numpy as np
from itertools import product
import unittest


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


def print_board(position):
    print('\n  |  a|b|c|d|e|f|g|h|j|k|l|m|n|o|p')
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


def print_data(data):
    for line, col in product(range(15), range(15)):
        if col == 14:
            end='\n'
        else:
            end='  '
        print(int(data[15 * line + col]), end=end)

