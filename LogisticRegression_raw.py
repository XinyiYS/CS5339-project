class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    # def __add_intercept(self, X):
        # intercept = np.ones((X.shape[0], 1))
        # return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def __forward(self,X,W):
        logits = np.dot(X, W)
        return __sigmoid(logits)[:,0]

    def __gradient(self, X, y, pred):
        return np.dot((pred - y), X).T/y.shape[0]

    def optimize_sgd(self, X_train, y_train, num_iter):

        for iter in range(num_iter):
            random_index = np.random.randint(0,len(X_train),1)
            X_batch, y_batch = X_train[random_index], y_train[random_index]
            pred = __forward(X_batch, self.W)
            dw = __gradient(X_batch, y_batch, pred).reshape(N_FEATURES,1)
            self.W -= self.lr * dw
            self.lr *= .9999

            if(self.verbose == True and i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print('loss: {self.__loss(h, y)}')

        return W


    def generate_adversarial(self,X_batch,y_batch,theta,epsilon):

        # random_batch_size = 20
        # random_batch_indices = np.random.choice(X_train.shape[0],random_batch_size,replace=False))
        # X_batch,y_batch = X_train[random_batch_indices],y_train[random_batch_indices]

        # derivative of cost function wrt x, which is a vector
        X,y = X_batch[0],y_batch[0]
        z = np.dot(X,W)
        sigmoid_z = _sigmoid(z)
        dx = (y * (sigmoid_z -1 )/(sigmoid_z)  + (1 - y) * (sigmoid_z)/(sigmoid_z-1) ) * theta
        X_adversarial = X + (epsilon * np.sign(dx)).reshape(X.shape[0])

        # plot_digit(X)
        # plot_digit(X_adversarial)
        X = X.reshape(1,-1)
        X_adversarial =X_adversarial.reshape(1,-1)
        W = W.reshape(-1,1)
        print("l1 norm of X - X_adversarial is ", np.linalg.norm(X-X_adversarial))
        print(__forward(X,W), __forward(X_adversarial,W))

        return X, X_adversarial

    def fit(self, X, y):
        # if self.fit_intercept:
            # X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.theta = optimize_sgd(self, X, y, self.num_iter)

        return self.theta



    def fit_adversarially(self, X, y, epsilon):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iter):

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            X_adversarial = X + epsilon * np.sign(gradient) # create the adversarial example

            z = np.dot(X_adversarial, self.theta) # fit the adversarial example
            h = self.__sigmoid(z)
            gradient = np.dot(X_adversarial.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X_adversarial, self.theta)
                h = self.__sigmoid(z)
                print('loss: {self.__loss(h, y)}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
