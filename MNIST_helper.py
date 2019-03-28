from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn.linear_model import LogisticRegression
import numpy as np
import time,os

def get_mnist():
    mndata = MNIST(os.path.join('MNIST')) # use this way if run locally, since MNIST is downloaded already
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    X_train,y_train = np.array(X_train).astype('float32')/255,np.array(y_train).astype('int64')
    X_test, y_test = np.array(X_test).astype('float32')/255,np.array(y_test).astype('int64')
    return X_train, y_train, X_test, y_test

mndata = MNIST(os.path.join('MNIST')) # use this way if run locally, since MNIST is downloaded already

def get_class_by_number(n):

    return images,labels
