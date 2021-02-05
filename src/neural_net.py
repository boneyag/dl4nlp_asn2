import numpy as np
import pickle

def sigmoid(z):
    """
    Sigmoid function.
    """
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of the sigmoid function.
    This is a useful during backpropagation.
    """

    return sigmoid(z) * (1 - sigmoid(z)) 

def gradient(y, y_hat):
    """
    Calculate gradient
    """
    pass

def linear_regression(A, W, b):
    """
    Calculate linear regression for for a layer.
    A - input matrix
    W - weights matrix
    b - bias scaler
    """

    return np.dot(W, A) + b
