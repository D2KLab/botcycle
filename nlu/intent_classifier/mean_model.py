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

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300 # spacy has glove with 300-dimensional embeddings

def one_layer():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(EMBEDDING_DIM,), dtype='float32')
    preds = Dense(len(intents), activation='softmax')(sequence_input)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

def two_layers():
    # sequence_input is a matrix of glove vectors (one for each input word)
    sequence_input = Input(
        shape=(EMBEDDING_DIM,), dtype='float32')
    hidden = Dense(200, activation='relu')(sequence_input)
    #model.add(Dropout(0.5))
    preds = Dense(len(intents), activation='softmax')(hidden)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

MODEL_NAME = os.environ['MODEL_NAME']
folder_name = MODEL_NAME + '__' + str(time.time())
models_available = {
    # very bad
    'mean_1l': one_layer,
    # a bit better
    'mean_2l': two_layers
}

def create_model():
    return models_available[MODEL_NAME]()

import model_utils

print('loading the data')
data_train = model_utils.load_data()

texts = []
labels = []

intents = model_utils.load_labels()
intents_lookup = model_utils.get_intents_lookup(intents)

inputs = np.zeros((len(data_train), EMBEDDING_DIM))
for idx, (text, intent) in enumerate(data_train):
    encoded = model_utils.encode_sentence(text)
    # sum up vectors
    inputs[idx] = encoded.mean(0)
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
