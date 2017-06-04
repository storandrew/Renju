
# coding: utf-8

# In[ ]:

import numpy as np
from skimage import transform


# In[ ]:

train = np.load('train.npy')
test = np.load('test.npy')


# In[ ]:

np.random.seed(42)
np.random.shuffle(train)


# In[ ]:

size = train.shape[0]
X = np.zeros(shape=(size, 64, 64, 1))
Y = np.zeros(shape=(size))

for i in range(train.shape[0]):
    image = transform.resize(train[i, 0], output_shape=(64, 64)).astype(np.float32)
    image -= 1
    image *= -1
    X[i] = image.reshape((64, 64, 1))
    Y[i] = train[i, 1]


# In[ ]:

np.save('X_final.npy', X)
np.save('Y_final.npy', Y)

