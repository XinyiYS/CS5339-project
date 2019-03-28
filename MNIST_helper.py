from sklearn.model_selection import train_test_split
from mnist import MNIST
import numpy as np
import time,os

class MNIST_helper:
    def __init__(self, dir):
        self.dir= dir
        self.mndata = MNIST(os.path.join(dir))
        mndata = MNIST(os.path.join('MNIST')) # use this way if run locally, since MNIST is downloaded already
        X_train, y_train = mndata.load_training()
        X_test, y_test = mndata.load_testing()
        self.X_test, self.y_test = np.array(X_test).astype('float32')/255,np.array(y_test).astype('int64')
        self.X_train, self.y_train = np.array(X_train).astype('float32')/255,np.array(y_train).astype('int64')

    def get_mnist(self,digits='all'):
        if digits=='all':
            return self.X_train,self.y_train,self.X_test, self.y_test
        else:
            digits = set(np.array(digits))
            indices_train = np.array([y in digits for y in self.y_train ])
            indices_test = np.array([y in digits for y in self.y_test ])
            return self.X_train[indices_train],self.y_train[indices_train],self.X_test[indices_test], self.y_test[indices_test]
