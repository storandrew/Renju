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
nb_classes = 225
nb_channels = 4
in_shape = (nb_rows, nb_rows, nb_channels)
train, validation = open_data('train.renju')

model = Sequential()

model.add(Flatten(input_shape=in_shape))
model.add(Dense(225, activation='softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy'
    , optimizer=Adadelta()
    , metrics=['accuracy']
    )

model.fit_generator(
    generate(train, mode='probability', batch_size=1024)
    , samples_per_epoch=1024000
    , nb_epoch=10
    , verbose=1
    , validation_data=generate(validation, mode='probability', batch_size=1024)
    , nb_val_samples=102400
    )

model.save('supervised_rollout.h5')
