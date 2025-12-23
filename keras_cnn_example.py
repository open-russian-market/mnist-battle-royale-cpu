import time

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Flatten
from sklearn.metrics import accuracy_score
import numpy as np

from utils.data_loader import get_mnist_data

# The keras version is 3.7.0.
# The tensorflow version is 2.18.0.
# Train time: 28.64 s
# Test score: 98.72 %

IMAGE_H, IMAGE_W, INPUT_CHANNELS = 28, 28, 1
NUMBER_OF_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 10


def get_model():
    # Input 28x28X1

    base_width = 8

    inputs = Input((IMAGE_H, IMAGE_W, INPUT_CHANNELS))
    x = inputs

    # Block 1
    x = Conv2D(base_width, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(base_width * 2, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(base_width * 4, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(base_width * 8, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(NUMBER_OF_CLASSES, (1, 1), activation=None, padding='same', name='block5_conv1')(x)

    x = Flatten()(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    #print(model.summary())

    return model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    X_train = X_train.reshape(-1, IMAGE_H, IMAGE_W, 1)
    X_test = X_test.reshape(-1, IMAGE_H, IMAGE_W, 1)
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