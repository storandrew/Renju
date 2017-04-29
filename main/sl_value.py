from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import load_model
from generator import open_data, generate
np.random.seed(42)

nb_rows = 15
nb_classes = 1
nb_channels = 4
in_shape = (nb_rows, nb_rows, nb_channels)
train, validation = open_data('train_no_draws.renju')

model = Sequential()

model.add(Convolution2D(32, 5, 5, input_shape=in_shape, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss='mse'
    , optimizer=Adadelta()
    , metrics=['accuracy']
    )

model.fit_generator(
    generate(train, mode='value', batch_size=1024)
    , samples_per_epoch=1024000
    , nb_epoch=5
    , verbose=1
    , validation_data=generate(validation, mode='value', batch_size=1024)
    , nb_val_samples=102400
    )

model.save('supervised_value.h5')
