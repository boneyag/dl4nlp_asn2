import numpy as np
from numpy import random
import pickle

import matplotlib
from numpy.core.fromnumeric import shape
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Sigmoid function.
    """
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    This is a useful during backpropagation.
    """

    return sigmoid(z) * (1 - sigmoid(z)) 

def linear_regression(A, W, b):
    """
    Calculate linear regression for for a layer.
    A - input matrix
    W - weights matrix
    b - bias scaler
    """

    return np.dot(W, A) + b

def get_accuracy(y_hat, y):
    """
    Calculate the accuracy on a set of predictions.
    """
    print(y_hat.shape)
    print(y.shape)
    return np.sum(y_hat == y, axis=1)/np.size(y, axis=1)

def training(X_train, y_train, X_val, y_val):
    """
    Training the neral network with mini-batch gradient descent
    Input layer size - 2000, hidden layer size - 200
    """

    il = 2000
    hl = 200
    batch = 20
    instances = X_train.shape[1]

    # initialize the parameters
    w1 = np.random.uniform(-0.5, 0.5, il*hl)
    w1 = w1.reshape((hl, il))
    b1 = np.random.uniform(-0.5, 0.5, hl)
    b1 = b1.reshape((hl, 1))
    w2 = np.random.uniform(-0.5, 0.5, hl)
    w2 = w2.reshape((1,hl))
    b2 = np.random.uniform(-0.5, 0.5, 1)
    b2 = b2.reshape((1,1))

    params = {
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2
    }

    alpha = 0.1 # fixed learning rate

    training_accuracy = list()
    validation_accuracy = list()

    model = {}

    for i in range(1,301,1):
        for j in range(1, instances, batch):

            # slice a mini-batch
            X = X_train[:, j:j+batch]
            y = y_train[:, j:j+batch]

            # print(X.shape)
            # print(y.shape)

            #forward
            z1 = linear_regression(X, params['w1'], params['b1'])
            A1 = sigmoid(z1)

            z2 = linear_regression(A1, params['w2'], params['b2'])
            A2 = sigmoid(z2)

            # backward
            dz2 = A2 - y
            dw2 = 1.0/batch * np.dot(dz2, A1.T)
            db2 = 1.0/batch * np.sum(dz2, axis=1, keepdims=True)

            dz1 = np.dot(params['w2'].T, dz2) * sigmoid_prime(z1)
            dw1 = 1.0/batch * np.dot(dz1, X.T)
            db1 = 1.0/batch * np.sum(dz1, axis=1, keepdims=True)

            # update params
            params['w1'] = params['w1'] - alpha * dw1
            params['b1'] = params['b1'] - alpha * db1
            params['w2'] = params['w2'] - alpha * dw2
            params['b2'] = params['b2'] - alpha * db2

       # calculate training and validation accuracy in each epoch

        tA1 = sigmoid(linear_regression(X_train, params['w1'], params['b1']))
        pred_t = sigmoid(linear_regression(tA1, params['w2'], params['b2']))
        
        pred_ct = (pred_t > 0.5).astype(int)
        
        training_accuracy.append(get_accuracy(pred_ct, y_train))

        vA1 = sigmoid(linear_regression(X_val, params['w1'], params['b1']))
        pred_v = sigmoid(linear_regression(vA1, params['w2'], params['b2']))
        
        pred_cv = (pred_v > 0.5 ).astype(int)
        
        curr_val_accuracy = get_accuracy(pred_cv, y_val)

        if i == 1:
            model['w1'] = params['w1']
            model['b1'] = params['b1']
            model['w2'] = params['w2']
            model['b2'] = params['b2']
        elif i > 1 and curr_val_accuracy > validation_accuracy[-1]:
            model['w1'] = params['w1']
            model['b1'] = params['b1']
            model['w2'] = params['w2']
            model['b2'] = params['b2']

        validation_accuracy.append(curr_val_accuracy)
        
    pickle.dump(model, open('../serialized/nn.pkl', 'wb'))
    pickle.dump((training_accuracy, validation_accuracy), open('../serialized/plot.pkl', 'wb'))

def test_model(X_test, y_test):
    """
    Test the final model's accuracy. Model was selected based on the highest accuracy on the vaidation set. 
    """
    model = pickle.load(open("../serialized/nn.pkl", "rb"))

    tA1 = sigmoid(linear_regression(X_test, model['w1'], model['b1']))
    y_expected = sigmoid(linear_regression(tA1, model['w2'], model['b2']))
    
    pred_c = (y_expected > 0.5).astype(int)
    # print(pred_c.shape)
    # print(pred_c[:10])
    # print(y_test.shape)
    # print(y_test[:10])

    accuracy = get_accuracy(pred_c, y_test)

    print(accuracy)

def plot(data1, data2):
    """
    Plot a graph.
    """

    plt.plot(data1, 'b-', label="training")
    plt.plot(data2, 'g-', label="validation")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()