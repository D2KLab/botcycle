import json
import os
import spacy
import time
import numpy as np

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.utils import plot_model

from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

# my modules
from data import loader
import utils

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ['DATASET']
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 40))
MODEL_OUTPUT_FOLDER = os.environ.get('MODEL_OUTPUT_FOLDER', '{}/models/{}/{}'.format(MY_PATH, DATASET, str(time.time())))

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300 # spacy has glove with 300-dimensional embeddings

def bidirectional_lstm(len_output):
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = Bidirectional(LSTM(100))(sequence_input)
    preds = Dense(len_output, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[utils.f1_score])

    """
    model.add(Bidirectional(LSTM(shape['nr_hidden'])))
    # dropout to avoid overfitting
    model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                metrics=['accuracy'])
    """

    return model

def lstm(len_output):
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = LSTM(200)(sequence_input)
    preds = Dense(len_output, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[utils.f1_score])

    return model

def gru(len_output):
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = GRU(200)(sequence_input)
    preds = Dense(len_output, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[utils.f1_score])

    return model

def bidirectional_gru(len_output):
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = Bidirectional(GRU(100))(sequence_input)
    preds = Dense(len_output, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[utils.f1_score])

    return model

# required, see values below
MODEL_NAME = os.environ['MODEL_NAME']
models_available = {
    # very bad
    'lstm': lstm,
    # very bad
    'gru': gru,
    # very good
    'bidirectional_lstm': bidirectional_lstm,
    'bidirectional_gru': bidirectional_gru
}

def create_model(len_output):
    return models_available[MODEL_NAME](len_output)

def prepare_inputs_and_outputs(dataset, intents_lookup):
    inputs = np.zeros((len(dataset), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    labels = np.zeros((len(dataset), len(intents_lookup)))
    for idx, value in enumerate(dataset):
        text = value['text']
        intent = value['intent']
        encoded = utils.encode_sentence(nlp, text)
        # copy the values, equivalent of padding
        inputs[idx,:encoded.shape[0],:encoded.shape[1]] = encoded[:MAX_SEQUENCE_LENGTH,:]
        # append the id of the intent
        labels[idx,intents_lookup[intent]] = 1

    print('Shape of inputs tensor:', inputs.shape)
    print('Shape of label tensor:', labels.shape)

    # shuffle the data
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    labels = labels[indices]

    return inputs,  labels

def train_and_evaluate(train, test, intents_lookup, save=False):
    validation_data = None
    train_inputs, train_labels = prepare_inputs_and_outputs(train, intents_lookup)
    if test:
        test_inputs, test_labels = prepare_inputs_and_outputs(test, intents_lookup)
        validation_data = test_inputs, test_labels

    print('Number of sentences for each intent, train and test')
    print([key for key in intents_lookup])
    print(train_labels.sum(axis=0))
    if test:
        print(test_labels.sum(axis=0))

    model = create_model(len(intents_lookup))
    # first iteration
    # model.summary()
    # this requires graphviz binaries also
    plot_model(model, to_file=MODEL_OUTPUT_FOLDER + '/model.png', show_shapes=True)

    history = model.fit(train_inputs, train_labels, validation_data=validation_data, epochs=MAX_ITERATIONS, batch_size=50)

    # keep only f1_scores
    history = {'train': np.array(history.history['f1_score']), 'test': np.array(history.history.get('val_f1_score', []))}


    # compute f1 score weighted by support
    y_pred_train = model.predict(train_inputs)
    f1_train = f1_score(train_labels.argmax(axis=1),
                y_pred_train.argmax(axis=1), average='weighted')
    if test:
        y_pred_test = model.predict(test_inputs)
        f1_test = f1_score(test_labels.argmax(axis=1),
                    y_pred_test.argmax(axis=1), average='weighted')
    else:
        f1_test = None
    
    # generate confusion matrix
    # confusion = utils.my_confusion_matrix(test_labels.argmax(
    #     axis=1), y_pred_test.argmax(axis=1), len(intents_lookup))

    print(f1_test, f1_train)
    if save:
        model.save(MODEL_OUTPUT_FOLDER + '/model.h5')
        stats = {}
        stats['model_name'] = MODEL_NAME
        stats['model'] = model.get_config()
        with open(MODEL_OUTPUT_FOLDER+'/stats.json', 'w+') as stats_file:
            json.dump(stats, stats_file)

    return history, f1_test, f1_train

np.random.seed(0)
nlp = spacy.load('en')
utils.main_flow_intents(DATASET, train_and_evaluate, MODEL_OUTPUT_FOLDER)