import time

import catboost
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from utils.data_loader import get_mnist_data

# The catboost version is 1.2.7.
# Train time: 839.37 s
# Test score: 97.03 %

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    print('-' * 60)
    print('The catboost version is {}.'.format(catboost.__version__))

    t0 = time.time()
    clf = CatBoostClassifier(random_state=0, verbose=False)
    clf.fit(X_train, y_train)
    print("Train time: %.2f s" % (time.time() - t0))

    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Test score: %.2f %%" % (score * 100.0))





