import numpy as np
from LogisticRegression_raw import LogisticRegression
from MNIST_helper import MNIST_helper

data_helper = MNIST_helper('MNIST')
X_train, y_train, X_test, y_test = data_helper.get_mnist([4,9])


lg = LogisticRegression(lr=0.05, num_iter=100, fit_intercept=False, verbose=False)
lg.fit(X_train,y_train)
