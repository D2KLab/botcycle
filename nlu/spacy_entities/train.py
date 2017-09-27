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

from data import loader

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ['DATASET']
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 1000))
SPACY_MODEL_NAME = os.environ.get('SPACY_MODEL_NAME', 'en')
DROP = float(os.environ.get('DROP', 0.9))
LEARN_RATE = float(os.environ.get('LEARN_RATE', 0.001))
MODEL_OUTPUT_FOLDER = os.environ.get('MODEL_OUTPUT_FOLDER', '{}/models/{}'.format(MY_PATH, DATASET))

def get_entities_lookup(entities):
    """From a list of entities gives back a dictionary that maps the value to the index in the original list"""
    entities_lookup = {}
    for index, value in enumerate(entities):
        entities_lookup[value] = index
    return entities_lookup

def train_and_evaluate(train, test, entities_lookup):
    # dimensions +1 because also no entity class (indexed last)
    extended_entities_lookup = entities_lookup.copy()
    extended_entities_lookup['NONE'] = len(entities_lookup)
    # collect in the history all the measures while training
    history = {'train': [], 'test': []}

    nlp = spacy.load(SPACY_MODEL_NAME)
    # passing None because don't want to save every fold model
    # TODO there
    train_ner(nlp, train, test, [key for key in entities_lookup], None, tot_iterations, drop, learn_rate)
    
    train_confusion_partial = eval_confusion(data[train], nlp, extended_entities_lookup)
    test_confusion_partial = eval_confusion(data[test], nlp, extended_entities_lookup)

    train_confusion_sum += train_confusion_partial
    test_confusion_sum += test_confusion_partial

    # free memory before death
    del nlp
    gc.collect()

    f1_train = f1_score(train_confusion_sum)
    f1_test = f1_score(test_confusion_sum)
    
    print('final confusion matrix:\n', test_confusion_sum)
    return test_confusion_sum, extended_entities_lookup, f1_test, f1_train

def eval_confusion(evaluation_data, nlp, extended_entities_lookup):
    confusion_sum = np.zeros((len(extended_entities_lookup), len(extended_entities_lookup)))
    for test_data in evaluation_data:
        # test_data is like {'text':'', 'entities':{'entity','value','start','end'}}
        doc = nlp(test_data['text'])
        # true_ents maps 'start_index:end_index' of entity to entity name, e.g. {'10:16': 'LOCATION'}
        true_ents = {'{}:{}'.format(true_ent['start'], true_ent['end']): true_ent['entity'].upper() for true_ent in test_data['entities']}
        
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
                print('wrong prediction in "' + str(doc) + '". "' + str(predicted_ent.text) + '" was classified as', predicted_class, 'but was', true_class)

        for false_negative in true_ents.values():
            print('false negative found: ' + false_negative)
            confusion_sum[extended_entities_lookup[false_negative], extended_entities_lookup['NONE']] += 1

        # now also add some NONE->NONE values, one for each sentence? TODO
        confusion_sum[extended_entities_lookup['NONE'], extended_entities_lookup['NONE']] += 1
    
    return confusion_sum

def train_ner(nlp, data, entity_names, output_directory, tot_iterations, drop, learn_rate):
    train_data = list(map(lambda x: (x['text'], list(map(lambda ent: (
        ent['start'], ent['end'], ent['entity'].upper()), x['entities']))), data))
    for entity_name in entity_names:
            nlp.entity.add_label(entity_name)

    # Add new words to vocab
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]
    random.seed(0)
    # You may need to change the learning rate. It's generally difficult to
    # guess what rate you should set, especially when you have limited data.
    nlp.entity.model.learn_rate = learn_rate

    # average of last iterations, for printing something
    tot_loss = 0.
    for itn in range(tot_iterations):
        random.shuffle(train_data)
        loss = 0.
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            nlp.tagger(doc)
            # As of 1.9, spaCy's parser now lets you supply a dropout probability
            # This might help the model generalize better from only a few
            # examples.
            loss += nlp.entity.update(doc, gold, drop=drop)

        tot_loss += loss / len(train_data)

        if loss == 0:
            break

        if (itn + 1) % 50 == 0:
            print('loss: ' + str(tot_loss / 50))
            tot_loss = 0.
    # This step averages the model's weights. This may or may not be good for
    # your situation --- it's empirical.
    nlp.end_training()

    # Save to directory
    if output_directory:
        nlp.save_to_directory(output_directory)

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


def main(learn_rate='0.001', drop='0.9', model_name='en', output_directory='models'):
    print("Loading initial model", model_name)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    data, entity_types, intent_types = loader.load_data('wit')
    entities = list(map(str.upper,entity_types))
    entities_lookup = get_entities_lookup(entity_types)

    if len(data) == 2:
        # only test, train in this order (alphabetical)
        test, train = data[0], data[1]
        result = train_and_evaluate(train, test, entities_lookup)
    else:
        # evaluate 5-fold
        for idx, test in enumerate(data):
            train = sum([example for index, example in enumerate(data) if index != idx], [])
            print('Running Fold', idx + 1, '/', n_folds, 'with', len(train), 'train samples and', len(test), 'test samples')
            result = train_and_evaluate(train, test, entities_lookup)
            # TODO average of folds

    # TODO plot the measures


    drop = float(drop)
    learn_rate = float(learn_rate)
    if True:
        test_confusion, extended_entities_lookup, f1_test, f1_train = kfold(model_name, n_folds, data, entities_lookup, tot_iterations, drop, learn_rate)
        plot_confusion(test_confusion, [key for key in extended_entities_lookup], output_directory + '/confusion_' + str(n_folds) + 'folds_' + str(tot_iterations) + 'iteration_' + str(drop) + 'drop_' + str(learn_rate) + 'learnrate')

        print(n_folds, 'folds', tot_iterations, 'iterations', learn_rate, 'learn_rate', drop, 'dropout', 'f1 on test data:', f1_test, 'f1 on train data:', f1_train)

    # now train on full data
    nlp = spacy.load('en')
    train_ner(nlp, data, entities, output_directory, tot_iterations, drop, learn_rate)

if __name__ == '__main__':
    import plac
    plac.call(main)
