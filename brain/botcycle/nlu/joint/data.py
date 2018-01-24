"""
Module with functions related to data loading and processing.

IMPORTANT: this file is hard-linked both at /nlu/joint/data.py and at /brain/botcycle/nlu/joint/data.py
"""

import json
import os
import random
import numpy as np


def flatten(list_of_lists):
    """Flattens from two-dimensional list to one-dimensional list"""
    return [item for sublist in list_of_lists for item in sublist]

def collapse_multi_turn_sessions(dataset):
    """Turns sessions into lists of messages with previous intent and previous bot turn (words and slot annotations)"""
    sessions = dataset['data']
    dataset['data'] = []
    # hold the previous intent value, initialized to some value (not important)
    previous_intent = dataset['meta']['intent_types'][0]
    previous_bot_turn = []
    previous_bot_slots = []
    for s in sessions:
        for m in s:
            #print('before')
            #print(m)
            if m['turn'] == 'b':
                # this is the bot turn
                previous_bot_turn = m['words']
                previous_bot_slots = m['slots']
            else:
                m['previous_intent'] = previous_intent
                m['bot_turn_actual_length'] = len(previous_bot_turn)
                m['words'] = previous_bot_turn + m['words']
                # TODO check this last modification also on slots
                m['slots'] = previous_bot_slots + m['slots']
                m['length'] += m['bot_turn_actual_length']
                dataset['data'].append(m)
            #print('after')
            #print(m['words'])
    return dataset

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

def spacy_wrapper(embedding_size, language, nlp, words_numpy):
    embeddings_values = np.zeros([words_numpy.shape[0], words_numpy.shape[1], embedding_size], dtype=np.float32)
    for j, column in enumerate(words_numpy.T):
        # rebuild the sentence
        words = [w.decode('utf-8') for w in column]
        real_length = words.index('<EOS>')
        # special value for EOS
        embeddings_values[real_length,j,:] = np.ones((embedding_size))
        # remove padding words, embedding values have already been initialized to zero
        words = words[:real_length]
        if language == 'it':
            # TODO handle correctly uppercase/lowercase
            #words = [w.lower() for w in words]
            pass
        # put back together the sentence in order to get the word embeddings with context (only for languages without vectors)
        # TODO skip this if always word vectors, since if word vectors are part of the model, they are fixed and can get them simply by doing lookup
        # unless contextual vectors can be built also when vectors are there
        sentence = ' '.join(words)
        if language == 'en' or language == 'it':
            # only make_doc instead of calling nlp, much faster
            doc = nlp.make_doc(sentence)
        else:
            # other languages don't have pretrained word embeddings but use context vectors, really slower
            doc = nlp(sentence)
        # now get the vectors for each token
        for i, w in enumerate(doc):
            if i < real_length:
                if i >= words_numpy.shape[0]:
                    print('out of length', w)
                    print(sentence)
                else:
                    if not w.has_vector:
                        # TODO if oov:
                        #   try lowercase
                        #print('word', w, 'does not have a vector')
                        punctuations = '.?!,;:-_()[]{}\''
                        # TODO handle OOV punctuation marks without special case
                        if language == 'it' and w.text in punctuations:
                            punct_idx = punctuations.index(w.text)
                            embeddings_values[i,j,:] = np.ones((embedding_size))*punct_idx+2
                    else:
                        embeddings_values[i,j,:] = w.vector
                
    return embeddings_values


def get_language_model_name(language):
    if language == 'en':
        return 'en_vectors_web_lg'
    if language == 'it':
        return 'it_vectors_wiki_lg'

    return language
