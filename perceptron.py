import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate  # Learning rate
        self.n_iters = n_iters  # Number of iterations over the training set
        self.activation_func = self._unit_step_func  # Activation function
        self.weights = None  # Weights after fitting
        self.bias = None  # Bias after fitting

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Main loop to fit the data to the perceptron model
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        # Activation function that returns 1 if x is positive, and -1 otherwise
        return np.where(x >= 0, 1, -1)
