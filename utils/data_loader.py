from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def get_mnist_data():
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    # X = X.astype(np.float32)
    y = y.astype(np.int32)

    X = X / 255.0

    print('np.min(X), np.max(X)', np.min(X), np.max(X))

    print('X.shape', X.shape)
    print('y.shape', y.shape)

    print('X.dtype', X.dtype)
    print('y.dtype', y.dtype)

    unique, counts = np.unique(y, return_counts=True)
    print('unique: counts', dict(zip(unique, counts)))

    n_train = 60000
    n_test = 10000
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=0
    )

    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)

    print('X_train.dtype', X_train.dtype)
    print('X_test.dtype', X_test.dtype)
    print('y_train.dtype', y_train.dtype)
    print('y_test.dtype', y_test.dtype)

    print('np.min(X_train), np.max(X_train)', np.min(X_train), np.max(X_train))
    print('np.min(X_test), np.max(X_test)', np.min(X_test), np.max(X_test))

    return X_train, X_test, y_train, y_test

