from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn.linear_model import LogisticRegression
import numpy as np
import time,os

class MNIST_helper:
    def __init__(self, dir):
        self.dir= dir
        self.mndata = MNIST(os.path.join(dir)
        mndata = MNIST(os.path.join('MNIST')) # use this way if run locally, since MNIST is downloaded already
        X_train, y_train = mndata.load_training()
        X_test, y_test = mndata.load_testing()
        self.X_test, self.y_test = np.array(X_test).astype('float32')/255,np.array(y_test).astype('int64')
        self.X_train, self.y_train = np.array(X_train).astype('float32')/255,np.array(y_train).astype('int64')

    def get_mnist(self):
        return self.X_train,self.y_train,self.X_test, self.y_test

    def get_mnist_by_digit(self,digit):
        train_indices,test_indices = self.y_train==n, self.y_test ==n
        return self.X_train[train_indices],self.y_train[train_indices],self.X_test[test_indices], self.y_test[test_indices]
