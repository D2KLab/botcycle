import tensorflow as tf
import numpy as np
import sys
import os
import spacy
from spacy.gold import iob_to_biluo, offsets_from_biluo_tags

from .model import RestoredModel
from .data import get_language_model_name

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get('DATASET', 'wit_it')

real_folder = MY_PATH + '/results/' + DATASET + '/'

def init():
    language = DATASET.split('_')[1]
    language_model_name = get_language_model_name(language)
    nlp = spacy.load(language_model_name)
    model = RestoredModel(real_folder, 300, language, nlp)

    return model, nlp


def process(model, nlp, sentence):
    doc = nlp.make_doc(line)
    words_true = [w.text for w in doc]
    length = len(words_true)
    words_true += ['<EOS>']
    words = words_true + ['<PAD>'] * (50-len(words_true))
    words = np.array(words)
    batch = [{
        'words': words,
        'length': length
    }]
    decoder_prediction, intent, intent_score = model.test(batch)
    # get the part that corresponds to words (truncate PAD and EOS)
    decoder_prediction = decoder_prediction[:length,0]
    #print(decoder_prediction, intent[0], intent_score)
    # clean up <EOS> and <PAD>
    decoder_prediction = [t if (t != '<EOS>' and t != '<PAD>') else 'O' for t in decoder_prediction]
    biluo_tags = iob_to_biluo(decoder_prediction)
    entities_offsets = offsets_from_biluo_tags(doc, biluo_tags)
    entities = []
    for ent in entities_offsets:
        e_parts = ent[2].split('.')
        if len(e_parts) > 1:
            # role.type
            entity = {'role': e_parts[0], 'type': e_parts[1]}
        else:
            entity = {'role': None, 'type': e_parts[0]}
        value = line[ent[0]: ent[1]]
        entities.append({'type': entity['type'], 'role': entity['role'], 'value': value})
    
    return {'type': intent, 'score': intent_score}, entities

if __name__ == '__main__':
    model, nlp = init()
    while True:
        line = input('> ')
        #words = np.tile('<PAD>', (50))
        print(process(model, nlp, line))