import spacy
import sys
import numpy as np
import operator

from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import preprocess_data

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

model = load_model('models/bidirectional_lstm/model.h5')
nlp = spacy.load('en')

print('Test your sentences.')
print('> ', end='', flush=True)

intents = preprocess_data.load_intents()

for line in sys.stdin:
    doc = nlp(line)
    embedding_matrix = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    for index, word in enumerate(doc):
        embedding_matrix[0][index] = word.vector
    prediction = model.predict(embedding_matrix)
    scores = {}
    for (x, y), score in np.ndenumerate(prediction):
        scores[intents[y]] = score

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_scores)

    print('> ', end='', flush=True)