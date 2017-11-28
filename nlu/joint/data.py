import json
import os
import random
import numpy as np


flatten = lambda l: [item for sublist in l for item in sublist]  # flatten from two-dimensional to one-dimensional
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

def load_data(dataset_name):
    """Returns data_splitted.

    data_splitted can be [fold1, fold2, fold3, fold4, fold5] or [test, train], look at the size"""
    path = 'data/' + dataset_name + '/preprocessed'

    fold_files = os.listdir(path)
    fold_files = sorted([f for f in fold_files if f.startswith('fold_')])

    data_splitted = []
    for file_name in fold_files:
        with open(path + '/' + file_name) as json_file:
            data_splitted.append(json.load(json_file))

    return data_splitted

def data_pipeline(data, length=50):
    # TODO use data in structured form instead of other arrays
    sin = []
    lengths = []
    sout = []
    intent = []
    for sample in data['data']:
        if len(sample['words']) < length:
            sample['words'].append('<EOS>')
            while len(sample['words']) < length:
                sample['words'].append('<PAD>')
        else:
            sample['words'] = sample['words'][:length]
            sample['words'][-1] = '<EOS>'

        sin.append(sample['words'])
        true_length = sample['length']
        lengths.append(true_length)

        if len(sample['slots']) < length:
            sample['slots'].append('<EOS>')
            while len(sample['slots']) < length:
                sample['slots'].append('<PAD>')
        else:
            sample['slots'] = sample['slots'][:length]
            sample['slots'][-1] = '<EOS>'
        sout.append(sample['slots'])
        intent.append(sample['intent'])
    
    #data = list(zip(sin, lengths, sout, intent))
    return data

def get_vocabularies(train_data):
    """
    collect the input vocabulary, the slot vocabulary and the intent vocabulary
    """
    # from a list of training examples, get three lists (columns)
    data = train_data['data']
    seq_in = [sample['words'] for sample in data]
    vocab = set(flatten(seq_in))
    # removing duplicated but keeping the order
    v = ['<PAD>','<SOS>', '<EOS>'] + list(vocab)
    vocab = sorted(set(v), key=lambda x: v.index(x))
    s = ['<PAD>', '<EOS>'] + train_data['meta']['slot_types']
    slot_tag = sorted(set(s), key=lambda x: s.index(x))
    intent_tag = set(train_data['meta']['intent_types'])

    return vocab, slot_tag, intent_tag


# TODO fix batch generation, the last one is skipped now
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        #print('returning',len(batch), 'samples')
        yield batch
