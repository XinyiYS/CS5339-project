class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train_sgd(self, X_train, y_train, l_rate, n_epoch):
    	coef = [0.0 for i in range(len(train[0]))]
    	for epoch in range(n_epoch):
    		for X,y in zip(X_train,y_train):
    			yhat = self.predict( X ,threshold = 0.5)
    			error = y - yhat
    			self.theta =  self.theta + l_rate * error * yhat * (1.0 - yhat)
    			# for i in range(len(row)-1):
    				# coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    	return

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print('loss: {self.__loss(h, y)}')

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
