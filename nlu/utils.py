import math
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import loader

def f1_score(y_true, y_pred):
    """
    supports = confusion.sum(axis=1)
    # TODO remove this ignore divide by 0, shouldn't happen
    with np.errstate(divide='ignore', invalid='ignore'):
        precisions = np.true_divide(tps, confusion.sum(axis=0))
        recalls = np.true_divide(tps, supports)
        f1s = 2*np.true_divide((precisions*recalls),(precisions+recalls))
        f1s[f1s == np.inf] = 0
        f1s = np.nan_to_num(f1s)
    f1 = np.average(f1s, weights=supports)
    """
    from keras import backend as K

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    supports = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    predict_distr = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    precisions = true_positives / predict_distr
    recalls = true_positives / supports
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    # get 0 instead of NaN
    f1_scores = tf.where(tf.is_nan(f1_scores), tf.zeros_like(f1_scores), f1_scores)
    f1 = K.sum(f1_scores * supports) / K.sum(supports)
    return f1

def plot_f1_history(file_name, train_f1_history, test_f1_history=np.array([])):
    plt.clf()
    plt.plot(train_f1_history)
    if test_f1_history.shape[0] > 0:
        plt.plot(test_f1_history)
        plt.legend(['train', 'test'], loc='lower right')
    else:
        plt.legend(['train'], loc='lower right')

    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epochs')
    print(file_name)
    plt.savefig(file_name)

def get_reverse_lookup(values):
    """From a list of values gives back a dictionary that maps each value to the index in the original list"""
    reverse_lookup = {}
    for index, value in enumerate(values):
        reverse_lookup[value] = index
    return reverse_lookup

def encode_sentence(nlp, sentence, embedding_dim=300):
    """convert from sentences to glove matrix"""
    # parse the sentence
    doc = nlp(sentence)
    result = np.zeros((len(doc), embedding_dim))
    for index, word in enumerate(doc):
        result[index] = word.vector
    
    return result

def my_confusion_matrix(y_true, y_pred, n_classes):
    """This function returns the confusion matrix tolerant to classes without true samples"""
    from scipy.sparse import coo_matrix
    CM = coo_matrix((np.ones(y_true.shape[0], dtype=np.int), (y_true, y_pred)),
                    shape=(n_classes, n_classes)
                    ).toarray()
    return CM

def main_flow_entities(dataset, callback, model_output_folder):
    return main_flow(dataset, callback, model_output_folder, 'entities')

def main_flow_intents(dataset, callback, model_output_folder):
    return main_flow(dataset, callback, model_output_folder, 'intents')

def main_flow(dataset, callback, model_output_folder, reverse_lookup_type):
    if not os.path.isdir(model_output_folder):
        os.makedirs(model_output_folder)
    print('loading the data')
    data, entity_types, intent_types = loader.load_data(dataset)
    if reverse_lookup_type is 'entities':
        reverse_lookup = get_reverse_lookup(entity_types)
    else:
        reverse_lookup = get_reverse_lookup(intent_types)
    
    if len(data) == 2:
        # only test, train in this order (alphabetical)
        test, train = data[0], data[1]
        print('Running with', len(train), 'train samples and', len(test), 'test samples')
        history, f1_test, f1_train = callback(train, test, reverse_lookup)
    else:
        # evaluate 5-fold
        histories = []
        for idx, test in enumerate(data):
            train = sum([example for index, example in enumerate(data) if index != idx], [])
            print('Running Fold', idx + 1, '/', len(data), 'with', len(train), 'train samples and', len(test), 'test samples')
            history_i, f1_test_i, f1_train_i = callback(train, test, reverse_lookup)
            histories.append(history_i)
        # do average of history on the folds
        h_f1_train = np.zeros((len(histories[0]['train'])))
        h_f1_test = np.zeros((len(histories[0]['train'])))
        for hist in histories:
            h_f1_train += hist['train']
            h_f1_test += hist['test']

        h_f1_train /= len(histories)
        h_f1_test /= len(histories)
        history = {'train': h_f1_train, 'test': h_f1_test}

    # plot the measures
    plot_f1_history(model_output_folder + '/f1_over_epochs.png', history['train'], history['test'])

    # now train of full dataset for top performances
    print('doing full train')
    train = sum([example for index, example in enumerate(data)], [])
    history, _, f1_train = callback(train, [], reverse_lookup, True)
