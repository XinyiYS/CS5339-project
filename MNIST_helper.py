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
            indices_train = np.array([y in digits for y in self.y_train ])
            indices_test = np.array([y in digits for y in self.y_test ])
            if len(digits) == 2:
                # one vs one
                digit_one, digit_two  = digits[0], digits[1]
                y_train, y_test = self.y_train[indices_train], self.y_test[indices_test]
                y_train[y_train== digit_one] = 1
                y_train[y_train  == digit_two] = 0
                y_test[y_test == digit_one] = 1
                y_test[y_test == digit_two] = 0
                return self.X_train[indices_train], y_train,\
                        self.X_test[indices_test], y_test

            else:
                # need to implement softmax via one-hot
                pass
                digits = set(np.array(digits))
                indices_train = np.array([y in digits for y in self.y_train ])
                indices_test = np.array([y in digits for y in self.y_test ])
                return self.X_train[indices_train],self.y_train[indices_train],self.X_test[indices_test], self.y_test[indices_test]
