import numpy as np

def open_data(name, split=0.9):
    lines = []
    with open(name) as raw_data:
        for line in raw_data:
            lines.append(line)

    train = lines[: int(split * len(lines))]
    validation = lines[int(split * len(lines)) :]
    return train, validation

def generate(data, batch_size, mode, nb_channels=4, nb_rows=15):
    if mode == 'probability':
        nb_classes = 225
    elif mode == 'value':
        nb_classes = 1
    X_train = np.zeros((batch_size, nb_rows, nb_rows, nb_channels))
    Y_train = np.zeros((batch_size, nb_classes))
    while 1:
        size = 0
        for line in data:
            field = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            white = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            black = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            is_white = 1
            moves = line.split()
            for move in moves[1:]:
                if size == batch_size:
                    yield X_train, Y_train
                    X_train = np.zeros((batch_size, nb_rows, nb_rows, nb_channels))
                    Y_train = np.zeros((batch_size, nb_classes))
                    size = 0
                x_coord, y_coord = ord(move[0]) - ord('a'), int(move[1:]) - 1
                if ord(move[0]) > ord('i'):
                    x_coord -= 1
                X_train[size, :, :, 0] = field
                X_train[size, :, :, 3] = np.ones((nb_rows, nb_rows))
                if is_white:
                    X_train[size, :, :, 1] = white
                    X_train[size, :, :, 2] = black
                else:
                    X_train[size, :, :, 1] = black
                    X_train[size, :, :, 2] = white
                if nb_classes == 225:
                    Y_train[size, 15 * x_coord + y_coord] = 1
                elif moves[0] == 'white' and is_white or moves[0] == 'black' and not is_white:
                    Y_train[size] = 1
                size += 1
                field[x_coord, y_coord] = 1
                if is_white:
                    white[x_coord, y_coord] = 1
                else:
                    black[x_coord, y_coord] = 1
                is_white = 1 - is_white

