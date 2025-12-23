import time

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.data_loader import get_mnist_data

# The scikit-learn version is 1.5.1.
# Train time: 6.83 s
# Test score: 96.80 %

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    print('-' * 60)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    t0 = time.time()
    clf = RandomForestClassifier(n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    print("Train time: %.2f s" % (time.time() - t0))

    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Test score: %.2f %%" % (score * 100.0))
