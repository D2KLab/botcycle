import tensorflow as tf
import numpy as np
import sys
import os
import spacy

from . import data
from .model import Model
from . import main

MY_PATH = os.path.dirname(os.path.abspath(__file__))

test_data, train_data = data.load_data('wit_en')
# fix the random seeds
#random_seed_init(len(test_data['data']))
training_samples = data.data_pipeline(train_data)
vocabs = data.get_vocabularies(training_samples)
model = main.get_model(vocabs, training_samples['meta']['tokenizer'], training_samples['meta']['language'])
global_init_op = tf.global_variables_initializer()
table_init_op = tf.tables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
# initialize the required parameters
sess.run(global_init_op)
sess.run(table_init_op)
# TODO restoring is not working!!! discover why
# look at http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver.restore(sess, MY_PATH + '/saved/model.ckpt')

# only tokenizer needed
nlp = spacy.load(training_samples['meta']['language'])

while True:
    line = input('> ')
    words = np.tile('<PAD>', (50))
    doc = nlp.make_doc(line)
    words_true = [w.text for w in doc]
    length = len(words_true)
    words[:length] = words_true
    words[length] = '<EOS>'
    batch = [{
        'words': words,
        'length': len(words),
        'slots': [],
        'intent': ''
    }]
    decoder_prediction, intent = model.step(sess, "test", batch)
    #print(batch)
    print([p for p in decoder_prediction[:,0]], intent[0])