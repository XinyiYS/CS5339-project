
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# from mnist import MNIST
import numpy as np
from sklearn.datasets import fetch_openml
from skorch import NeuralNetClassifier
import time,os


# mnist = fetch_openml('mnist_784', cache=False)
# X = mnist.data.astype('float32')
# y = mnist.target.astype('int64')
# X /= 255.0

torch.manual_seed(0);
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# XCnn = X.reshape(-1, 1, 28, 28)

from MNIST_help import get_mnist

X_train, y_train, X_test, y_test = get_mnist()
XCnn_train = X_train.reshape(-1, 1, 28, 28)
XCnn_test = X_test.reshape(-1, 1, 28, 28)



class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) # flatten over channel, height and width = 1600
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=10,
    lr=1,
    optimizer=torch.optim.Adadelta,
    device=device,
)

cnn.fit(XCnn_train, y_train)
cnn_pred = cnn.predict(XCnn_test)
print(np.mean(cnn_pred == y_test))
