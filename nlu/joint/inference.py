import tensorflow as tf
import numpy as np
import sys
import os
import spacy

from . import data
from .model import Model, RestoredModel
from .embeddings import get_language_model_name
from . import main

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get('DATASET', 'wit_it')

real_folder = MY_PATH + '/results/last/' + DATASET + '/'

test_data, train_data = data.load_data(DATASET)
# fix the random seeds
#random_seed_init(len(test_data['data']))
training_samples = data.data_pipeline(train_data)

# only tokenizer needed
language = training_samples['meta']['language']
language_model_name = get_language_model_name(language)
nlp = spacy.load(language_model_name)
model = RestoredModel(real_folder, 300, language, nlp)

while True:
    line = input('> ')
    #words = np.tile('<PAD>', (50))
    doc = nlp.make_doc(line)
    words_true = [w.text for w in doc]
    length = len(words_true)
    words_true += ['<EOS>']
    words = words_true + ['<PAD>'] * (50-len(words_true))
    words = np.array(words)
    #words = np.lib.pad(words_true , (0,50-len(words_true)), mode='constant', constant_values='<PAD>')
    #words[:length] = words_true
    #words[length] = '<EOS>'
    batch = [{
        'words': words,
        'length': len(words)
    }]
    print(batch)
    decoder_prediction, intent = model.test(batch)
    print([p for p in decoder_prediction[:,0]], intent[0])