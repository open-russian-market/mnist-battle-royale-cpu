import time

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, LSTM
from sklearn.metrics import accuracy_score
import numpy as np

from utils.data_loader import get_mnist_data

# The keras version is 3.7.0.
# The tensorflow version is 2.18.0.
# Train time: 85.68 s
# Test score: 98.79 %

IMAGE_H, IMAGE_W, INPUT_CHANNELS = 28, 28, 1
NUMBER_OF_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 10


def get_model():
    # Input 28x28

    inputs = Input((IMAGE_H, IMAGE_W)) # (batch, timesteps, feature)
    x = inputs

    x = LSTM(128)(x)
    x = Dense(NUMBER_OF_CLASSES)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model.summary())

    return model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    X_train = X_train.reshape(-1, IMAGE_H, IMAGE_W)
    X_test = X_test.reshape(-1, IMAGE_H, IMAGE_W)
    y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)

    print('-' * 60)
    print('The keras version is {}.'.format(keras.__version__))
    print('The tensorflow version is {}.'.format(tf.__version__))

    t0 = time.time()
    model = get_model()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
    print("Train time: %.2f s" % (time.time() - t0))

    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    score = accuracy_score(y_test, y_pred)
    print("Test score: %.2f %%" % (score * 100.0))