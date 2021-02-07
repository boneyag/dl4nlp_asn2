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
    
    return np.sum(y_hat == y, axis=0)/np.size(y, axis=1)

def forward_propagation(A, params):
    z1 = linear_regression(A, params['w1'], params['b1'])
    A1 = sigmoid(z1)

    z2 = linear_regression(A1, params['w2'], params['b2'])
    A2 = sigmoid(z2)

    return z1, A1, z2, A2

def neural_net(X, y, params, alpha):
    """
    The function generate two layer neural network with sigmoid activation function for all neurons.
    Input layer size - 2000, hidden layer size - 200
    Cost function - cross entropy
    """
    m1 = params['w1'].shape[1]
    m2 = params['w2'].shape[1]
    # forward
    z1, A1, z2, A2 = forward_propagation(X, params)

    # backward
    dz2 = A2 - y
    dw2 = 1.0/m2 * np.dot(dz2, A1.T)
    db2 = 1.0/m2 *np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(params['w2'].T, dz2) * sigmoid_prime(z1)
    dw1 = 1.0/m1 * np.dot(dz1, X.T)
    db1 = 1.0/m1 * np.sum(dz1, axis=1, keepdims=True)

    # update weights 
    params['w1'] -= alpha * dw1
    params['b1'] -= alpha * db1
    params['w2'] -= alpha * dw2
    params['b2'] -= alpha * db2

    return params

def training(X_train, y_train, X_val, y_val):
    """
    Training the neral network with mini-batch gradient descent
    Input layer size - 2000, hidden layer size - 200
    """

    l1 = 2000
    h1 = 200

    # initialize the parameters
    w1 = np.random.uniform(-0.5, 0.5, l1*h1)
    w1 = np.reshape(w1, (h1, l1))
    b1 = random.uniform(-0.5, 0.5)
    w2 = np.random.uniform(-0.5, 0.5, h1)
    w2 = np.reshape(w2, (1,h1))
    b2 = random.uniform(-0.5, 0.5)

    params = {
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2
    }

    alpha = 0.1 # fixed learning rate

    training_accuracy = list()
    validation_accuracy = list()

    model = {'w1': params['w1'],
             'b1': params['b2'],
             'w2': params['w2'],
             'b2': params['b2']
             }

    for i in range(1,301,1):
        current_instance = 0
        for j in range(1, int(l1/20)+1, 1):

            # slice a mini-batch
            X = X_train[:, current_instance:current_instance+20]
            y = y_train[:, current_instance:current_instance+20]
            current_instance += 20

            # print(X.shape)
            # print(y.shape)
            params = neural_net(X, y, params, alpha)

            
        _, _, _, pred_t = forward_propagation(X_train, params)
        
        pred_ct = (pred_t > 0.5).astype(int)
        
        training_accuracy.append(get_accuracy(pred_ct, y_train)[0])

        _, _, _, pred_v = forward_propagation(X_val, params)
        
        pred_cv = (pred_v > 0.5 ).astype(int)
        
        curr_val_accuracy = get_accuracy(pred_cv, y_val)[0]

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
        
    pickle.dump(params, open('../serialized/nn.pkl', 'wb'))
    pickle.dump((training_accuracy, validation_accuracy), open('../serialized/plot.pkl', 'wb'))

def test_model(X_test, y_test):
    """
    Test the final model's accuracy. Model was selected based on the highest accuracy on the vaidation set. 
    """
    model = pickle.load(open("../serialized/nn.pkl", "rb"))

    _, _, _, y_expected = forward_propagation(X_test, model)
    m = y_expected.shape[1]
    y_expected = y_expected.reshape((m,))
    pred_c = (y_expected.T > 0.5).astype(int)
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