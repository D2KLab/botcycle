import json
import os
import random
import numpy as np


def flatten(list_of_lists):
    """Flattens from two-dimensional list to one-dimensional list"""
    return [item for sublist in list_of_lists for item in sublist]


def load_data(dataset_name, mode='measures'):
    """Loads the dataset and returns it.
    
    if mode='measures' (default), returns [test_data, train_data]
    
    if mode='runtime', returns [None, all the data together], to do a full training to be used at runtime"""
    path = 'data/' + dataset_name + '/preprocessed'

    fold_files = os.listdir(path)
    fold_files = sorted([f for f in fold_files if f.startswith('fold_')])
    final_test = 'final_test.json'

    data_splitted = []
    for file_name in fold_files:
        with open(path + '/' + file_name) as json_file:
            data_splitted.append(json.load(json_file))

    if mode == 'measures':
        return data_splitted
    elif mode == 'runtime':
        with open(path + '/' + final_test) as json_file:
            result = json.load(json_file)
        for split in data_splitted:
            result['data'].extend(split['data'])
        return None, result
    else:
        raise ValueError('mode unsupported:' + mode)


def adjust_sequences(data, length=50):
    """Fixes the input and output sequences in length, adding padding or truncating if necessary"""
    for sample in data['data']:
        # adjust the sequence of input words
        if len(sample['words']) < length:
            # add <EOS> and <PAD> if sentence is shorter than maximum length
            sample['words'].append('<EOS>')
            while len(sample['words']) < length:
                sample['words'].append('<PAD>')
        else:
            # otherwise truncate and add <EOS> at last position
            sample['words'] = sample['words'][:length]
            sample['words'][-1] = '<EOS>'

        # adjust in the same way the sequence of output slots
        if len(sample['slots']) < length:
            sample['slots'].append('<EOS>')
            while len(sample['slots']) < length:
                sample['slots'].append('<PAD>')
        else:
            sample['slots'] = sample['slots'][:length]
            sample['slots'][-1] = '<EOS>'

    return data


def get_vocabularies(train_data):
    """Collect the input vocabulary, the slot vocabulary and the intent vocabulary"""
    # from a list of training examples, get three lists (columns)
    data = train_data['data']
    seq_in = [sample['words'] for sample in data]
    vocab = flatten(seq_in)
    # removing duplicated but keeping the order
    v = ['<PAD>', '<SOS>', '<EOS>'] + vocab
    vocab = sorted(set(v), key=lambda x: v.index(x))
    s = ['<PAD>', '<EOS>'] + train_data['meta']['slot_types']
    slot_tag = sorted(set(s), key=lambda x: s.index(x))
    i = train_data['meta']['intent_types']
    intent_tag = sorted(set(i), key=lambda x: i.index(x))

    return vocab, slot_tag, intent_tag


def get_batch(batch_size, train_data):
    """Returns iteratively a batch of specified size on the data. The last batch can be smaller if the total size is not multiple of the batch"""
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while sindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        #print('returning', len(batch), 'samples')
        yield batch
