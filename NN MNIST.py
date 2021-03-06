import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
from mnist import MNIST
import os

MNIST_helper= MNIST_helper('MNIST')
X_train, y_train, X_test, y_test = MNIST_helper.get_mnist()

torch.manual_seed(0);
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X_train.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(y_train))

class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

from skorch import NeuralNetClassifier

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
)

net.fit(X_train, y_train);
predicted = net.predict(X_test)
print(np.mean(predicted == y_test))
