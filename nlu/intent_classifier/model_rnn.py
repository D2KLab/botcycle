import os
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

import model_utils

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300 # spacy has glove with 300-dimensional embeddings

def bidirectional_lstm():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = Bidirectional(LSTM(100))(sequence_input)
    preds = Dense(len(intents), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    """
    model.add(Bidirectional(LSTM(shape['nr_hidden'])))
    # dropout to avoid overfitting
    model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                metrics=['accuracy'])
    """

    return model

def lstm():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = LSTM(200)(sequence_input)
    preds = Dense(len(intents), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

def gru():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = GRU(200)(sequence_input)
    preds = Dense(len(intents), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

def bidirectional_gru():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='float32')
    l_lstm = Bidirectional(GRU(100))(sequence_input)
    preds = Dense(len(intents), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

# required, see values below
MODEL_NAME = os.environ['MODEL_NAME']
folder_name = MODEL_NAME + '__' + str(time.time())
models_available = {
    # very bad
    'lstm': lstm,
    # very bad
    'gru': gru,
    # very good
    'bidirectional_lstm': bidirectional_lstm,
    'bidirectional_gru': bidirectional_gru
}

def create_model():
    return models_available[MODEL_NAME]()

print('loading the data')
data_train = model_utils.load_data()

texts = []
labels = []

intents = model_utils.load_labels()
intents_lookup = model_utils.get_intents_lookup(intents)

inputs = np.zeros((len(data_train), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for idx, (text, intent) in enumerate(data_train):
    encoded = model_utils.encode_sentence(text)
    # copy the values, equivalent of padding
    inputs[idx,:encoded.shape[0],:encoded.shape[1]] = encoded[:MAX_SEQUENCE_LENGTH,:]
    # append the id of the intent
    labels.append(intents_lookup[intent])

# now from intent_1 to [0,1,0,...]
labels = to_categorical(np.asarray(labels))
print('Shape of inputs tensor:', inputs.shape)
print('Shape of label tensor:', labels.shape)

# shuffle the data
indices = np.arange(inputs.shape[0])
np.random.shuffle(indices)
inputs = inputs[indices]
labels = labels[indices]

print('Number of sentences for each intent')
print(intents)
print(labels.sum(axis=0))

n_folds = 5
f1 = model_utils.kfold(create_model, n_folds, inputs, labels, intents, folder_name)

model_utils.save_full_train(create_model, inputs, labels, folder_name, {'f1': f1})
