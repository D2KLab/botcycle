import os
import json
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model

import spacy

import preprocess_data

EMBEDDING_DIM = 300 # spacy has glove with 300-dimensional embeddings
MODELS_PATH = 'models/'

nlp = spacy.load('en')

def load_data():
    return preprocess_data.get_train_data(preprocess_data.load_expressions())

def load_labels():
    return preprocess_data.load_intents()

def get_intents_lookup(intents):
    """From a list of intents gives back a dictionary that maps the value to the index in the original list"""
    intents_lookup = {}
    for index, value in enumerate(intents):
        intents_lookup[value] = index
    return intents_lookup

def encode_sentence(sentence):
    """convert from sentences to glove matrix"""
    # parse the sentence
    doc = nlp(sentence)
    result = np.zeros((len(doc), EMBEDDING_DIM))
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

def plot_confusion(confusion, label_values, path):
    # to avoid overflow of canvas, labels are numbers
    df_cm = pd.DataFrame(confusion, index=range(len(label_values)), columns=range(len(label_values)))
    fig = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    fig.set_xlabel('predict')
    fig.set_ylabel('actual')
    fig.get_figure().savefig(path + '.png')
    plt.clf()

def kfold(create_model_function, n_folds, data, labels, label_names, model_name):
    model_path = MODELS_PATH + model_name + '/'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    """data must be a numpy array, one element for each input value"""
    # skf will profide indices to iterate over in each fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    f1_scores = np.zeros((n_folds))
    confusion_sum = np.zeros((labels.shape[1], labels.shape[1]))

    history = []

    for i, (train, test) in enumerate(skf.split(np.zeros((data.shape[0],)), np.zeros((data.shape[0],)))):
        model = create_model_function()
        epochs = 50
        if i == 0:
            # first iteration
            model.summary()
            # this requires graphviz binaries also
            plot_model(model, to_file=model_path + 'model.png', show_shapes=True)

        print("Running Fold", i + 1, "/", n_folds)

        history_i = model.fit(data[train], labels[train], validation_data=(
            data[test], labels[test]), epochs=epochs, batch_size=50)

        history.append(history_i)

        # generate confusion matrix
        y_pred = model.predict(data[test])
        confusion = my_confusion_matrix(labels[test].argmax(
            axis=1), y_pred.argmax(axis=1), labels.shape[1])

        # compute f1 score weighted by support
        f1 = f1_score(labels[test].argmax(axis=1),
                    y_pred.argmax(axis=1), average='weighted')
        print('f1 at fold ' + str(i + 1) + ': ' + str(f1))
        
        f1_scores[i] = f1
        confusion_sum = np.add(confusion_sum, confusion)

        plot_confusion(confusion, label_names, model_path + 'confusion_iteration_' + str(i + 1))

    f1_mean = f1_scores.mean()
    print('mean f1 score: ' + str(f1_mean))
    plot_confusion(confusion_sum, label_names, model_path + 'confusion_sum')

    # do average of history on the folds
    acc = np.zeros((epochs))
    val_acc = np.zeros((epochs))
    for h in history:
        acc += h.history['acc']
        val_acc += h.history['val_acc']

    acc /= len(history)
    val_acc /= len(history)

    # summarize history for accuracy
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(model_path + 'acccuracy.png')

    return f1_mean

def save_full_train(create_model_function, inputs, labels, model_name, stats):
    model_path = MODELS_PATH + model_name + '/'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    print("Now training on full dataset, no validation")
    model = create_model_function()
    # TODO get nb_epoch from caller
    #model.fit(inputs, labels, nb_epoch=10, batch_size=50)

    model.save(model_path + 'model.h5')
    stats['model_name'] = model_name
    stats['model'] = model.get_config()

    with open(model_path+'/stats.json', 'w+') as stats_file:
        json.dump(stats, stats_file)