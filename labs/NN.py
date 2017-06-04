
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta

np.random.seed(42)


# In[ ]:

X = np.load('X_final.npy')
Y = np.load('Y_final.npy')
size = Y.shape[0]
from skimage.filters import threshold_otsu
for i in range(size):
    image = X[i, :, :].reshape((64, 64))
    thresh = threshold_otsu(image)
    binary = image > thresh
    X[i] = binary.reshape((64, 64, 1))


# In[ ]:

map_from, map_to = {}, {}
for n, label in enumerate(set(Y)):
    map_from[label] = n
    map_to[n] = label

for i in range(size):
    Y[i] = map_from[Y[i]]

nb_classes = 500
Y = np_utils.to_categorical(Y, nb_classes)


# In[ ]:

split = 0.8
X_train = X[: round(size * split), :]
X_val = X[round(size * split) : , :]

Y_train = Y[: round(size * split)]
Y_val = Y[round(size * split) :]


# In[ ]:

model = Sequential()

model.add(Convolution2D(64, 5, 5, input_shape=(64, 64, 1), border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])


# In[ ]:

model.fit(X_train, Y_train, batch_size=128, nb_epoch=10,
          verbose=2, validation_data=(X_val, Y_val))


# In[ ]:

model.fit(X_train, Y_train, batch_size=256, nb_epoch=10,
          verbose=2, validation_data=(X_val, Y_val))


# In[ ]:

model.fit(X_train, Y_train, batch_size=512, nb_epoch=10,
          verbose=2, validation_data=(X_val, Y_val))


# In[ ]:

model.fit(X_train, Y_train, batch_size=768, nb_epoch=50,
          verbose=2, validation_data=(X_val, Y_val))


# In[ ]:

test = np.load('test.npy')
size = test.shape[0]


# In[ ]:

X_test = []
for i in range(size):
    X_test.append(transform.resize(test[i], (64, 64)))
del test


# In[ ]:

X_test = np.array(X_test)
X_test -= 1
X_test *= -1
X_test = np.expand_dims(X_test, axis=3)

for i in range(X_test.shape[0]):
    image = X_test[i, :, :].reshape((64, 64))
    thresh = threshold_otsu(image)
    binary = image > thresh
    X_test[i] = binary.reshape((64, 64, 1))


# In[ ]:

Y = model.predict_classes(X_test, verbose=0)
for i in range(size):
    Y[i] = map_to[Y[i]]
s = pandas.Series(Y, index=range(1, size+1))
s.to_csv('test_final.csv', index=True, sep=',', header=['Category'], index_label='Id')


# In[ ]:



