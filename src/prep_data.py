import os
from bs4 import BeautifulSoup
import numpy as np
import random
import pickle

def create_vocab(pos_list, neg_list):
    """
    Read the training data set and create vocabulary
    """
    vocab = {}

    tr_pos_dir = os.scandir('../data/train/pos')

    for item in tr_pos_dir:
        if item.is_file() and item.name in pos_list:
            with open(item) as f:
                content = f.read()
                content = preprocess_text(content)
                
                for w in content:
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
               

    tr_neg_dir = os.scandir('../data/train/neg')

    for item in tr_neg_dir:
        if item.is_file() and item.name in neg_list:
            with open(item) as f:
                content = f.read()
                content = preprocess_text(content)
                
                for w in content:
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 1

    sorted_vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    frequent_words = [w for (w,c) in sorted_vocab[:2000]]   

    word2id = {w: (index+1) for (index, w) in enumerate(frequent_words)}
    id2word = {index: w for (w, index) in enumerate(frequent_words)}

    vocab = {'word2id': word2id, 'id2word': id2word}

    # print(list(word2id.items())[:10])
    # print(list(id2word.items())[:10])
    pickle.dump(vocab, open("../serialized/vocab.pkl", "wb"))

def split_train_val():
    """
    Split a 2500 files each from train--pos and train--neg to create a validation set.
    """

    pos_file = list()
    neg_file = list()

    tr_pos_dir = os.scandir('../data/train/pos')

    i = 0

    for item in tr_pos_dir:
        if item.is_file():
            pos_file.append(item.name)

        i += 1

    tr_neg_dir = os.scandir('../data/train/neg')

    i = 0

    for item in tr_neg_dir:
        if item.is_file():
            neg_file.append(item.name)

        i += 1

    random.shuffle(pos_file)
    random.shuffle(neg_file)

    tr_pos_files = pos_file[:10000]
    val_pos_files = pos_file[10000:12500]
    tr_neg_files = neg_file[0:10000]
    val_neg_files = neg_file[10000:125000]

    train_val = {
        'tr_pos' : tr_pos_files,
        'tr_neg' : tr_neg_files,
        'val_pos' : val_pos_files,
        'val_neg' : val_neg_files
    }

    return train_val


def create_data(vocab, file_name_dict):
    """
    Read trainig and testing data dirs and split training data to training a validation set. 
    """

    X_train = np.zeros((2000,20000))
    X_val = np.zeros((2000,5000))
    X_test = np.zeros((2000,25000))

    tr_pos_dir = os.scandir('../data/train/pos')

    i = 0
    j = 0
    for item in tr_pos_dir:
        if item.is_file():
            if item.name in file_name_dict['tr_pos']:
                with open(item) as f:
                    content = f.read()
                    words = preprocess_text(content)
                    unique_words = list(set(words))

                    for word in unique_words:
                        if word in vocab['id2word'].keys():
                            X_train[vocab['id2word'][word], i] = 1
    
                i += 1
            
            if item.name in file_name_dict['val_pos']:
                with open(item) as f:
                    content = f.read()
                    words = preprocess_text(content)
                    unique_words = list(set(words))

                    for word in unique_words:
                        if word in vocab['id2word'].keys():
                            X_val[vocab['id2word'][word], j] = 1
                            

                j += 1

        # if i > 2 or j > 2:
        #     break

    tr_neg_dir = os.scandir('../data/train/neg')

    for item in tr_neg_dir:
        if item.is_file():
            if item.name in file_name_dict['tr_neg']:
                with open(item) as f:
                    content = f.read()
                    words = preprocess_text(content)
                    unique_words = list(set(words))

                    for word in unique_words:
                        if word in vocab['id2word'].keys():
                            X_train[vocab['id2word'][word], i] = 1
                i += 1
            
            if item.name in file_name_dict['val_neg']:
                with open(item) as f:
                    content = f.read()
                    words = preprocess_text(content)
                    unique_words = list(set(words))

                    for word in unique_words:
                        if word in vocab['id2word'].keys():
                            X_val[vocab['id2word'][word], j] = 1
            
                j += 1

    y_train = [1] * 10000 + [0] * 10000
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (1, 20000))

    np.random.seed(314)
    np.random.shuffle(X_train.T)
    np.random.seed(314)
    np.random.shuffle(y_train)

    y_val = [1] * 2500 + [0] * 2500
    y_val = np.array(y_val)
    y_val = np.reshape(y_val, (1, 5000))

    test_pos_dir = os.scandir('../data/test/pos')

    i = 0

    for item in test_pos_dir:
        if item.is_file():
            with open(item) as f:
                content = f.read()
                words = preprocess_text(content)
                unique_words = list(set(words))
                
                for word in unique_words:
                    if word in vocab['id2word'].keys():
                        X_test[vocab['id2word'][word], i] = 1

    test_neg_dir = os.scandir('../data/test/neg')

    for item in test_neg_dir:
        if item.is_file():
            with open(item) as f:
                content = f.read()
                words = preprocess_text(content)
                unique_words = list(set(words))
                
                for word in unique_words:
                    if word in vocab['id2word'].keys():
                        X_test[vocab['id2word'][word], i] = 1

    y_text = [1] * 12500 + [0] * 12500
    y_test = np.array(y_text)
    y_test = np.reshape(y_test, (1, 25000))

    pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), open('../serialized/data.pkl', 'wb'))

def preprocess_text(comment):
    """
    Clean the text by removing html tags. The turn all words to lower case 
    """

    soup = BeautifulSoup(comment, 'lxml')

    text_only = soup.get_text()

    lowered_split_text = text_only.strip().lower().split()

    # processed_text = ''

    # for w in lowered_split_text:
    #     processed_text += w + ' '

    # return processed_text.strip()
    return lowered_split_text
