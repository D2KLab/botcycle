# coding=utf-8
# @author: cer

import random
import numpy as np


flatten = lambda l: [item for sublist in l for item in sublist]  # flatten from two-dimensional to one-dimensional
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def data_pipeline(data, length=50):
    data = [t[:-1] for t in data]  # Removed'\n'
    # One line of data like this：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # divide into [the words of the original sentence， annotated sequence，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # remove BOS and EOS, and remove the corresponding annotation sequence corresponding label
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    lengths = []
    sout = []
    # padding，end of original sequence and label sequence +<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)
        true_length = temp.index("<EOS>")
        lengths.append(true_length)

        temp = seq_out[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)
        data = list(zip(sin, lengths, sout, intent))
    return data

def get_vocabularies(train_data):
    """
    collect the input vocabulary, the slot vocabulary and the intent vocabulary
    """
    # from a list of training examples, get three lists (columns)
    seq_in, _, seq_out, intent = list(zip(*train_data))
    vocab = set(flatten(seq_in))
    # removing duplicated but keeping the order
    v = ['<PAD>','<SOS>', '<EOS>'] + list(vocab)
    vocab = sorted(set(v), key=lambda x: v.index(x))
    s = ['<PAD>', '<EOS>'] + flatten(seq_out)
    slot_tag = sorted(set(s), key=lambda x: s.index(x))
    intent_tag = set(intent)

    return vocab, slot_tag, intent_tag


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
