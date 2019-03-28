from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn.linear_model import LogisticRegression
import numpy as np
mndata = MNIST('./MNIST')
images, labels = mndata.load_training()
test_images,test_labels = mndata.load_testing()

def get_class_by_number(n):

    return images,labels
