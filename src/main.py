import pickle

from prep_data import split_train_val
from prep_data import create_vocab
from prep_data import create_data

def main():
    # only need to create the vocabulary once

    # tr_val_dict = split_train_val()
    # create_vocab(tr_val_dict['tr_pos'], tr_val_dict['tr_neg'])

    # pickle.dump(tr_val_dict, open('../serialized/file_names.pkl', 'wb'))

    # only need to prepare data once

    vocab = pickle.load(open('../serialized/vocab.pkl', 'rb'))
    tr_val_dict = pickle.load(open('../serialized/file_names.pkl', 'rb'))

    create_data(vocab, tr_val_dict)

    # only need to train once

    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open('../serialized/data.pkl', 'rb'))
    
    
    

if __name__ == '__main__':
    main()