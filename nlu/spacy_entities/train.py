#!/usr/bin/env python
# coding: utf8
"""
This code trains the NER with entities from the exported wit_data.
The entity types found in the wit_data are added to an existing pre-trained NER
model ('en').

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training the model, it is saved to a directory.
Documentation:
* Training the Named Entity Recognizer: https://spacy.io/docs/usage/train-ner
* Saving and loading models: https://spacy.io/docs/usage/saving-loading

Example adapted from https://github.com/explosion/spaCy/blob/master/examples/training/train_new_entity_type.py
"""
from __future__ import unicode_literals, print_function
import gc
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import random

import spacy
from spacy.gold import GoldParse
from spacy.tagger import Tagger

# my modules
from data import loader
import utils

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ['DATASET']
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 1000))
SPACY_MODEL_NAME = os.environ.get('SPACY_MODEL_NAME', 'en')
DROP = float(os.environ.get('DROP', 0.9))
LEARN_RATE = float(os.environ.get('LEARN_RATE', 0.001))
MODEL_OUTPUT_FOLDER = os.environ.get('MODEL_OUTPUT_FOLDER', '{}/models/{}'.format(MY_PATH, DATASET))
VERBOSE = os.environ.get('VERBOSE', False)

def json2spacy_ent(json_data):
    return list(map(lambda x: (x['text'], list(map(lambda ent: (
        ent['start'], ent['end'], ent['type']), x['entities']))), json_data))

def train_and_evaluate(train, test, entities_lookup, save=False):
    # dimensions +1 because also no entity class (indexed last)
    extended_entities_lookup = entities_lookup.copy()
    extended_entities_lookup['NONE'] = len(entities_lookup)
    # collect in the history all the measures while training
    history = {'train': [], 'test': []}

    nlp = spacy.load(SPACY_MODEL_NAME)

    train_data = json2spacy_ent(train)
    test_data = json2spacy_ent(test)
    for entity_name in entities_lookup:
            nlp.entity.add_label(entity_name)

    # Add new words to vocab
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]
    random.seed(0)
    # You may need to change the learning rate. It's generally difficult to
    # guess what rate you should set, especially when you have limited data.
    nlp.entity.model.learn_rate = LEARN_RATE

    for itn in range(MAX_ITERATIONS):
        random.shuffle(train_data)
        loss = 0.
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            nlp.tagger(doc)
            # As of 1.9, spaCy's parser now lets you supply a dropout probability
            # This might help the model generalize better from only a few
            # examples.
            loss += nlp.entity.update(doc, gold, drop=DROP)

        if loss == 0:
            break

        # every k iterations evaluate on train and test
        if (itn + 1) % 1 == 0:
            train_confusion = eval_confusion(train, nlp, extended_entities_lookup)
            f1 = f1_score(train_confusion)
            print('train iteration', itn, 'f1', f1)
            history['train'].append(f1)
            if test:
                test_confusion = eval_confusion(test, nlp, extended_entities_lookup)
                f1 = f1_score(test_confusion)
                print('test iteration', itn, 'f1', f1)
                history['test'].append(f1)


    # This step averages the model's weights. This may or may not be good for
    # your situation --- it's empirical.
    nlp.end_training()

    # final evaluation
    train_confusion = eval_confusion(train, nlp, extended_entities_lookup)
    f1_train = f1_score(train_confusion)
    print('train final', 'f1', f1_train)
    if test:
        test_confusion = eval_confusion(test, nlp, extended_entities_lookup)
        f1_test = f1_score(test_confusion)
        print('test final', 'f1', f1_test)
    else:
        f1_test = None

    # Save to directory
    if save and MODEL_OUTPUT_FOLDER:
        nlp.save_to_directory(MODEL_OUTPUT_FOLDER)

    # convert to numpy
    history['test'] = np.array(history['test'])
    history['train'] = np.array(history['train'])

    # free memory before death
    del nlp
    gc.collect()

    return history, f1_test, f1_train

def eval_confusion(evaluation_data, nlp, extended_entities_lookup):
    confusion_sum = np.zeros((len(extended_entities_lookup), len(extended_entities_lookup)))
    for test_data in evaluation_data:
        # test_data is like {'text':'', 'entities':{'type','value','start','end'}}
        doc = nlp(test_data['text'])
        # true_ents maps 'start_index:end_index' of entity to entity type, e.g. {'10:16': 'LOCATION'}
        true_ents = {'{}:{}'.format(true_ent['start'], true_ent['end']): true_ent['type'] for true_ent in test_data['entities']}
        
        for predicted_ent in doc.ents:
            # on match an entry from true_ents is removed (see below computation of false negatives)
            true_ent = true_ents.pop('{}:{}'.format(predicted_ent.start_char, predicted_ent.end_char), 'NONE')
            # the fallback parameter is needed in case unexpected types of entities are found
            predicted_class = extended_entities_lookup.get(predicted_ent.label_, extended_entities_lookup['NONE'])
            true_class = extended_entities_lookup[true_ent]
            # actual class indexes the rows while predicted class indexes the columns
            confusion_sum[true_class, predicted_class] += 1
            if predicted_class is not true_class:
                # TODO careful to boundaries
                if VERBOSE:
                    print('wrong prediction in "' + str(doc) + '". "' + str(predicted_ent.text) + '" was classified as', predicted_class, 'but was', true_class)

        for false_negative in true_ents.values():
            if VERBOSE:
                print('false negative found: ' + false_negative)
            confusion_sum[extended_entities_lookup[false_negative], extended_entities_lookup['NONE']] += 1

        # now also add some NONE->NONE values, one for each sentence? TODO
        confusion_sum[extended_entities_lookup['NONE'], extended_entities_lookup['NONE']] += 1
    
    return confusion_sum


def f1_score(confusion):
    tps = np.diagonal(confusion)
    supports = confusion.sum(axis=1)
    # TODO remove this ignore divide by 0, shouldn't happen
    with np.errstate(divide='ignore', invalid='ignore'):
        precisions = np.true_divide(tps, confusion.sum(axis=0))
        recalls = np.true_divide(tps, supports)
        f1s = 2*np.true_divide((precisions*recalls),(precisions+recalls))
        f1s[f1s == np.inf] = 0
        f1s = np.nan_to_num(f1s)
    f1 = np.average(f1s, weights=supports)
    return f1

# TODO remove duplicated code, same as intent model utils
def plot_confusion(confusion, label_values, path):
    df_cm = pd.DataFrame(confusion, index=label_values, columns=label_values)
    df_cm.columns.name = 'predict'
    df_cm.index.name = 'actual'
    #sn.set(font_scale=1.4)  # for label size
    fig = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    fig.get_figure().savefig(path + '.png')
    plt.clf()


def main():
    utils.main_flow_entities(DATASET, train_and_evaluate, MODEL_OUTPUT_FOLDER)

if __name__ == '__main__':
    import plac
    plac.call(main)
