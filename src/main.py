import pickle
import numpy as np
from numpy import random

from prep_data import split_train_val
from prep_data import create_vocab
from prep_data import create_data
from neural_net import training
from neural_net import test_model
from neural_net import plot

def main():
    # only need to create the vocabulary once

    # split the files for traing and validation
    tr_val_dict = split_train_val()

    # create the vocabulary for training data files
    create_vocab(tr_val_dict['tr_pos'], tr_val_dict['tr_neg'])

    pickle.dump(tr_val_dict, open('../serialized/file_names.pkl', 'wb'))

    # only need to prepare data once

    vocab = pickle.load(open('../serialized/vocab.pkl', 'rb'))
    tr_val_dict = pickle.load(open('../serialized/file_names.pkl', 'rb'))

    # create vectorized representation of data
    create_data(vocab, tr_val_dict)

    # training and testing

    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open('../serialized/data.pkl', 'rb'))
   
    # train the model
    training(X_train, y_train, X_val, y_val)

    [tr_ac, vl_ac] = pickle.load(open("../serialized/plot.pkl", "rb"))

    # plot training and validation accuracy
    plot(tr_ac, vl_ac)

    # calculate test accuracy
    test_model(X_test, y_test)
    
    

if __name__ == '__main__':
    main()