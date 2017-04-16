from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import load_model

np.random.seed(42)

nb_classes = 225
nb_rows = 15
nb_channels = 4
in_shape = (nb_rows, nb_rows, nb_channels)
batch_size = 1024

lines = []
with open('train.renju') as raw_data:
    for line in raw_data:
        lines.append(line)

split = 0.9
train = lines[: int(split * len(lines))]
validation = lines[int(split * len(lines)) :]

def generate(data):
    X_train = np.zeros((batch_size, nb_rows, nb_rows, nb_channels))
    Y_train = np.zeros((batch_size, nb_classes))
    while 1:
        size = 0
        for line in data:
            field = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            white = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            black = np.zeros((nb_rows, nb_rows), dtype=np.float32)
            is_white = 1
            for move in line.split()[1:]:
                if size == batch_size:
                    yield X_train, Y_train
                    X_train = np.zeros((batch_size, nb_rows, nb_rows, nb_channels))
                    Y_train = np.zeros((batch_size, nb_classes))
                    size = 0
                x_coord, y_coord = ord(move[0]) - ord('a'), int(move[1:]) - 1
                X_train[size, :, :, 0] = field
                X_train[size, :, :, 3] = np.ones((nb_rows, nb_rows))
                if is_white:
                    X_train[size, :, :, 1] = white
                    X_train[size, :, :, 2] = black
                else:
                    X_train[size, :, :, 1] = black
                    X_train[size, :, :, 2] = white
                Y_train[size, 15 * x_coord + y_coord] = 1
                size += 1
                field[x_coord, y_coord] = 1
                if is_white:
                    white[x_coord, y_coord] = 1
                else:
                    black[x_coord, y_coord] = 1
                is_white = 1 - is_white

model = Sequential()

model.add(Convolution2D(64, 3, 3, input_shape=in_shape, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy'
    , optimizer=Adadelta()
    , metrics=['accuracy']
    )

model.fit_generator(
    generate(train)
    , samples_per_epoch=1024000
    , nb_epoch=15
    , verbose=1
    , validation_data=generate(validation)
    , nb_val_samples=102400
    )

model.save('supervised_policy.h5')
