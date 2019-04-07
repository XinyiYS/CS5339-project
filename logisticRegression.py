# adversarially train with SGD
# generate adversarial examples
from sklearn import datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from MNIST_helper import MNIST_helper

data_helper = MNIST_helper('MNIST')
X_train, y_train, X_test, y_test = data_helper.get_mnist([4,9])

BATCH_SIZE = 1
STEPS = 1000
LEARNING_RATE = 0.05
N_FEATURES =X_train.shape[1]

def _sigmoid(logits):
    return 1/(1 + np.exp(-logits))

def forward(X, W):
    logits = np.dot(X, W)
    return _sigmoid(logits)[:,0]

def gradient(X, y, pred):
    return np.dot((pred - y), X).T/y.shape[0]

def predict(X,W):
    probs = np.array(forward(X,W))
    predictions = (probs > 0.5).astype('int64')
    return predictions


def plot_digit(some_digit):

    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image, cmap = 'viridis', interpolation = "nearest")
    plt.axis("off")
    plt.show()

def save_digit_plot(some_digit,digit,adversarial=False):
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image, cmap = 'viridis', interpolation = "nearest")
    plt.axis("off")
    if adversarial:
        plt.savefig('{}_adversarial.png'.format(str(digit)))
    else:
        plt.savefig('{}.png'.format(str(digit),adversarial))

    # plt.show()

    return

# def get_next_batch():
    # return df.iloc[start:end,:][features], df.iloc[start:end]['y']

# initialize
W = np.random.random([N_FEATURES, 1])

for step in range(STEPS):
    random_index = np.random.randint(0,len(X_train),1)
    X_batch, y_batch = X_train[random_index], y_train[random_index]
    pred = forward(X_batch, W)
    dw = gradient(X_batch, y_batch, pred).reshape(N_FEATURES,1)
    W -= LEARNING_RATE*dw
    LEARNING_RATE *= .9999

    def generate_adversarial(epsilon,W,data_index):

        # derivative of cost function wrt x, which is a vector
        X,y = X_train[data_index],y_train[data_index]
        z = np.dot(X,W)
        sigmoid_z = _sigmoid(z)
        dx = (y * (sigmoid_z -1 )/(sigmoid_z)  + (1 - y) * (sigmoid_z)/(sigmoid_z-1) ) * W

        X_adversarial = X + (epsilon * np.sign(dx)).reshape(X.shape[0])
        save_digit_plot(X,y_train[data_index])
        save_digit_plot(X_adversarial,y_train[data_index],True)

        X = X.reshape(1,-1)
        X_adversarial =X_adversarial.reshape(1,-1)
        W = W.reshape(-1,1)
        print("l1 norm of X - X_adversarial is ", np.linalg.norm(X-X_adversarial))
        print(forward(X,W), forward(X_adversarial,W))

        # print((X,W), predict(X_adversarial,W))
        return X, X_adversarial

    if step == STEPS - 1:

        # generate_adversarial(0.04,W,0)
        # generate_adversarial(0.04,W,1)
        generate_adversarial(-0.04,W,-2)


preds = predict(X_test,W)
print((preds==y_test).mean())


# train l1 norm, generate adversarial examples
# train l2 norm, generate adversarial examples
# train no regularisaition, generate adversarial examples
# train adversarially, generate adversarial examples

# test all models on all adversarial examples
