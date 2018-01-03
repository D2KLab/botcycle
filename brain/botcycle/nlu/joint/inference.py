import tensorflow as tf
import numpy as np
import sys
import os
import spacy
from spacy.gold import iob_to_biluo, offsets_from_biluo_tags

from .model import RestoredModel
from .data import get_language_model_name

MY_PATH = os.path.dirname(os.path.abspath(__file__))


class NeuralNetWrapper(object):
    def __init__(self, language, dataset_name):
        real_folder = MY_PATH + '/results/' + dataset_name + '/'
        language_model_name = get_language_model_name(language)
        self.nlp = spacy.load(language_model_name)
        self.model = RestoredModel(real_folder, 300, language, self.nlp)


    def process(self, line, intent_treshold_score=0.5):
        doc = self.nlp.make_doc(line)
        words_true = [w.text for w in doc]
        length = len(words_true)
        words_true += ['<EOS>']
        words = words_true + ['<PAD>'] * (50-len(words_true))
        words = np.array(words)
        batch = [{
            'words': words,
            'length': length
        }]
        decoder_prediction, intent, intent_score = self.model.test(batch)
        # batch only contains one element
        intent = intent[0]
        intent_score = intent_score[0]
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
            entities.append({'_entity': entity['type'],
                'role': entity['role'],
                'value': value,
                '_body': value,
                '_start': ent[0],
                '_end': ent[1]
            })

        # now convert to the same format as wit.ai, applying the treshold
        if intent_score < intent_treshold_score:
            intent_result = None
        else:
            intent_result = {'confidence': intent_score, 'value': intent}
        
        entities_result = {}
        for ent in entities:
            if ent['role']:
                entities_result[ent['role']] = ent
            else:
                entities_result[ent['type']] = ent
        
        return intent_result, entities_result

if __name__ == '__main__':
    dataset = os.environ.get('DATASET', 'wit_it')
    language = dataset.split('_')[1]
    nlu = NeuralNetWrapper(language, dataset)
    while True:
        line = input('> ')
        print(nlu.process(line))