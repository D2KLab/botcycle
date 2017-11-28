import os
import spacy
import sys
import numpy as np
import operator

from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data import loader
import utils

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

DATASET = os.environ['DATASET']
LANGUAGE = os.environ.get('LANGUAGE', 'en')
LANG_MODEL_PATH = os.environ.get('LANG_MODEL_PATH', None)
INTENT_MODEL_PATH = os.environ.get('INTENT_MODEL_PATH') + '/model.h5'

model = load_model(INTENT_MODEL_PATH, custom_objects={'f1_score': utils.f1_score})
nlp = utils.get_nlp(LANGUAGE, LANG_MODEL_PATH)

print('Test your sentences.')
print('> ', end='', flush=True)

_, _, intents = loader.load_data_old(DATASET)

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

    for ent in doc.ents:
        print('entity: ', ent.label_, 'value:', ent.text)

    print('> ', end='', flush=True)