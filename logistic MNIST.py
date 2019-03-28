
from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn.linear_model import LogisticRegression,SGDClassifier
import numpy as np
import time,os
from LogisticRegression_raw import LogisticRegression as LR_raw
from MNIST_helper import MNIST_helper


def train_MNIST_all():
    data_helper = MNIST_helper('MNIST')
    X_train, y_train, X_test, y_test = data_helper.get_mnist()
    print(X_train.shape,X_test.shape)

    # import LogisticRegression_raw
    logreg_SGD = SGDClassifier(loss='log',max_iter=400)
    start_time = time.time()
    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg_SGD.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    preds = logreg_SGD.predict(X_test)
    print((preds == y_test).mean())

def train_MNIST_one_v_one(d1,d2):
    data_helper = MNIST_helper('MNIST')
    X_train, y_train, X_test, y_test = data_helper.get_mnist([d1,d2])
    # import LogisticRegression_raw
    logreg_SGD = SGDClassifier(loss='log',max_iter=1)
    start_time = time.time()
    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg_SGD.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    preds = logreg_SGD.predict(X_test)
    print((preds == y_test).mean())


    coefs = logreg_SGD.coef_
    print(coefs)
    def create_adversarial(epsilon = 1/256):
        X_train[:10] +=  epsilon * np.sign()
        return

    return logreg_SGD

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return (sigmoid(z) - 1) / sigmoid(z)

def objective_prime_wrt_X(theta,y,X,bias=1):
    z = -y* np.dot(theta,x)+bias
    sigmoid_prime(z)


    return

def logistic_regression_clf_objective(theta,y,x,bias=1):
    sigmoid(-y* np.dot(theta,x)+bias)



    return


train_MNIST_one_v_one(1,7)
exit()

logreg_raw = LR_raw(lr=0.1, num_iter=100, fit_intercept=True, verbose=False)




logreg = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial',max_iter=100)
start_time = time.time()
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
preds = logreg.predict(X_test)
print((preds == y_test).mean())
