import time

import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from utils.data_loader import get_mnist_data

# The xgboost version is 2.1.3.
# Train time: 71.51 s
# Test score: 97.46 %

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    print('-' * 60)
    print('The xgboost version is {}.'.format(xgboost.__version__))

    t0 = time.time()
    clf = XGBClassifier(random_state=0)
    clf.fit(X_train, y_train)
    print("Train time: %.2f s" % (time.time() - t0))

    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Test score: %.2f %%" % (score * 100.0))


