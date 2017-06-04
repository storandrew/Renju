import numpy as np

empty =  0
white = 2
black = 1

class InvalidMoveError(Exception):
    pass

class Board(object):
    def __init__(self, height, width):
        self.history = 1
        self.height = int(height)
        self.width = int(width)
        self.shape = (self.height, self.width)
        self.board = np.zeros(self.shape, dtype='int8')

        self.reset()

    def reset(self):
        self.board[:] = empty
        self.moves_left = self.height * self.width
        self.in_turn = black
        if hasattr(self, 'lastmove'):
            del self.lastmove
        self.log = []

    def __getitem__(self, key):
        return self.board[key]

    def __setitem__(self, key, value):
        if value == self.in_turn and self[key] == empty:
            self.in_turn = 3 - self.in_turn
            assert self.moves_left > 0
            self.moves_left -= 1
            self.board[key] = value
            self.lastmove = key
            self.log.append(tuple(key))

        else: # invalid move
            if   self[key] != empty:
                raise InvalidMoveError('Position %s is already taken' % ((key),))
            elif self.in_turn == white:
                raise InvalidMoveError('white is in turn')
            elif self.in_turn == black:
                raise InvalidMoveError('black is in turn')
            else:
                raise RuntimeError('FATAL ERROR!')

    def full(self):
        if self.moves_left:
            return False
        else:
            return True

    def get_column(self, y, x, length=5):
        line = np.empty(length, dtype='int8')
        for i in range(length):
            line[i] = self[y+i,x]
        return line, [(y+i,x) for i in range(length)]

    def get_row(self, y, x, length=5):
        line = np.empty(length, dtype='int8')
        for i in range(length):
            line[i] = self[y,x+i]
        return line, [(y,x+i) for i in range(length)]

    def get_diagonal_upleft_to_lowright(self, y, x, length=5):
        line = np.empty(length, dtype='int8')
        for i in range(length):
            line[i] = self[y+i,x+i]
        return line, [(y+i,x+i) for i in range(length)]

    def get_diagonal_lowleft_to_upright(self, y, x, length=5):
        line = np.empty(length, dtype='int8')
        if y < length - 1:
            raise IndexError
        for i in range(length):
            line[i] = self[y-i,x+i]
        return line, [(y-i,x+i) for i in range(length)]

    def winner(self):
        for i in range(self.height):
            for j in range(self.width):
                for getter_function in (self.get_row
                        , self.get_column
                        , self.get_diagonal_lowleft_to_upright
                        , self.get_diagonal_upleft_to_lowright):
                    try:
                        line, positions = getter_function(i,j)
                    except IndexError:
                        continue
                    if line.prod() == 1 or line.prod() == 32:
                        return line[0], positions
        return None, []
