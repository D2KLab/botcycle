"""
This module preprocesses the datasets in equivalent formats
"""
import json
import os
import re
import numpy as np
import spacy
from spacy.gold import biluo_tags_from_offsets
from sklearn.model_selection import StratifiedShuffleSplit


def atis_preprocess_old():
    """
    preprocesses the atis dataset, taking as source the files atis.test.w-intent.iob and
    atis.train.w-intent.iob

    Produces in output the files entity_types.json, fold_train.json, fold_test.json

    For joint NLU use atis_preprocess()
    """
    with open('atis/source/atis.test.w-intent.iob') as txt_file:
        test_set = txt_file.readlines()
    with open('atis/source/atis.train.w-intent.iob') as txt_file:
        train_set = txt_file.readlines()
    train_tagged, entity_types, intent_types = atis_lines_to_json(train_set)
    test_tagged, test_entity_types, test_intent_types = atis_lines_to_json(test_set)

    # some entities and intents may appear only in test
    entity_types.update(test_entity_types)
    intent_types.update(test_intent_types)
    entity_types = list(sorted(entity_types))
    intent_types = list(sorted(intent_types))

    if not os.path.exists('atis/preprocessed_old'):
        os.makedirs('atis/preprocessed_old')

    with open('atis/preprocessed_old/intent_types.json', 'w') as outfile:
        json.dump(intent_types, outfile)

    with open('atis/preprocessed_old/entity_types.json', 'w') as outfile:
        json.dump(entity_types, outfile)

    with open('atis/preprocessed_old/fold_train.json', 'w') as outfile:
        json.dump(train_tagged, outfile)

    with open('atis/preprocessed_old/fold_test.json', 'w') as outfile:
        json.dump(test_tagged, outfile)


def atis_lines_to_json(content):
    """Transforms the content (list of lines) in json,
    detecting entity start and end indexes in sentences.
    Returns the tagged dataset, the enitity_types and the intent_types
    
    Used in the old preprocessing"""

    result = []

    entity_types = set()
    intent_types = set()

    for line in content:
        element = {}
        start_text_idx = line.find('BOS ') + 4
        end_text_idx = line.find('EOS', start_text_idx)
        text = line[start_text_idx:end_text_idx]
        text = text.strip()
        element['text'] = text
        start_annotations_idx = line.find('\t') + 1
        annotations = line[start_annotations_idx:]
        annotations = annotations.split()
        entities_tags = annotations[1:-1]
        intent = annotations[-1]
        # multi-intent is not taken into consideration. Its value will be the concatenation of intents using '#'
        intent_types.add(intent)
        element['intent'] = intent
        # chunks are defined by the space, IOB notations correspond to this split
        chunks = text[:start_annotations_idx - 1].split()
        entities = []
        state = 'O'
        entity = {}
        for idx, tag in enumerate(entities_tags):
            tag = tag.split('-')
            next_state = tag[0]
            if len(tag) == 2:
                simple_tag = tag[1]
                if next_state == 'B':
                    if state == 'B':
                        # close previous entity
                        entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
                        entity['value'] = element['text'][entity['start']:entity['end']]
                        entities.append(entity)
                    # beginning of new entity
                    entity = {'type': simple_tag, 'start': sum(
                        map(len, chunks[:idx])) + idx}
                    entity_types.add(simple_tag)

            if next_state == 'O' and state != 'O':
                # end of entity inside the sentence
                entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
                entity['value'] = element['text'][entity['start']:entity['end']]
                entities.append(entity)
                entity = {}

            # update state
            state = next_state

        if state != 'O':
            # last entity at the end of the sentence
            idx = len(entities_tags)
            entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
            entity['value'] = element['text'][entity['start']:entity['end']]
            entities.append(entity)
            entity = {}

        element['entities'] = entities
        result.append(element)

    return result, entity_types, intent_types


def wit_preprocess_old(path):
    """Preprocesses the wit.ai dataset from the folder path passed as parameter.
    To download the updated dataset, use the download.sh script.
    Saves the tagged dataset, the enitity_types and the intent_types
    
    For joint NLU use wit_preprocess()"""
    path_source = path + '/source'
    enitites_path = '{}/entities'.format(path_source)

    with open(enitites_path + '/intent.json') as json_file:
        intents = json.load(json_file)
    
    intent_types = list(map(lambda val: val['value'], intents['data']['values']))

    with open('{}/expressions.json'.format(path_source)) as json_file:
        expressions = json.load(json_file)

    dataset, entity_types = wit_get_normalized_data(expressions)

    # perform the split on 5 folds
    dataset = np.array(dataset)
    # initialize the random generator seed to the size of the dataset, just to
    # make it split always the same
    np.random.seed(dataset.size)
    np.random.shuffle(dataset)
    fold_size = len(dataset) // 5
    folds = [dataset[:fold_size], dataset[fold_size:2 * fold_size],
             dataset[2 * fold_size:3 * fold_size], dataset[3 * fold_size:4 * fold_size], dataset[4 * fold_size:]]

    if not os.path.exists('{}/preprocessed_old'.format(path)):
        os.makedirs('{}/preprocessed_old'.format(path))

    for idx, fold in enumerate(folds):
        with open('{}/preprocessed_old/fold_{}.json'.format(path, idx + 1), 'w') as outfile:
            json.dump(fold.tolist(), outfile)

    with open('{}/preprocessed_old/intent_types.json'.format(path), 'w') as outfile:
        json.dump(intent_types, outfile)

    with open('{}/preprocessed_old/entity_types.json'.format(path), 'w') as outfile:
        json.dump(entity_types, outfile)


def wit_get_normalized_data(expressions):
    """Returns a list of objects like `{'text': SENTENCE, 'intent': ,
    'entities': [{'entity': (role.)?ENTITY_NAME, 'value': ENTITY_VALUE, 'start': INT, 'end', INT}]}`
    
    followed by the entity types
    
    Is part of the old preprocessing"""
    entity_types = set()
    items = expressions['data']
    results = []
    for item in items:
        result = {'text': item['text'], 'intent': None, 'entities': []}
        for e_or_i in item['entities']:
            if e_or_i['entity'] == 'intent':
                result['intent'] = e_or_i['value'].strip('"')
            else:
                entity_type = e_or_i['entity']
                if 'role' in e_or_i:
                    entity_type = e_or_i['role'] + '.' + entity_type
                entity_types.add(entity_type)
                entity = {'type': entity_type, 'value': e_or_i['value'].strip('"'), 'start': e_or_i['start'], 'end': e_or_i['end']}
                result['entities'].append(entity)

        results.append(result)

    return results, list(sorted(entity_types))


"""
Methods below are for the preprocessing for joint task. Methods above are only for disjointed intent and entity tasks
"""

def atis_preprocess():
    """"Preprocesses the ATIS dataset for joint NLU"""
    # atis-2.train and atis-2.dev summed together make atis.train. atis.test is independent
    with open('atis/source/atis-2.train.w-intent.iob') as txt_file:
        train_set_raw = txt_file.readlines()
    with open('atis/source/atis-2.dev.w-intent.iob') as txt_file:
        dev_set_raw = txt_file.readlines()
    with open('atis/source/atis.test.w-intent.iob') as txt_file:
        test_set_raw = txt_file.readlines()

    train_set = iob_lines_to_structured_iob(train_set_raw)
    dev_set = iob_lines_to_structured_iob(dev_set_raw)
    test_set = iob_lines_to_structured_iob(test_set_raw)

    if not os.path.exists('atis/preprocessed'):
        os.makedirs('atis/preprocessed')

    with open('atis/preprocessed/fold_train.json', 'w') as outfile:
        json.dump(train_set, outfile)

    with open('atis/preprocessed/fold_test.json', 'w') as outfile:
        json.dump(dev_set, outfile)

    with open('atis/preprocessed/final_test.json', 'w') as outfile:
        json.dump(test_set, outfile)


def nlu_benchmark_preprocess(nlp):
    """Preprocess the nlu-benchmark dataset for joint NLU"""
    path = 'nlu-benchmark/2017-06-custom-intent-engines/'
    # the data is splitted by intent using different folders
    intent_folders = os.listdir(path)
    train_samples = []
    test_samples = []
    train_slot_types = set()
    test_slot_types = set()
    intents = set()
    for intent_type in intent_folders:
        intent_path = path + intent_type
        if os.path.isdir(intent_path):
            print('train ' + intent_type)
            with open(intent_path + '/train_' + intent_type + '_full.json') as json_file:
                train_json = json.load(json_file)
                train_iob, slot_types = nlu_benchmark_to_structured_iob(train_json, intent_type, nlp)
                train_slot_types.update(slot_types)
            print('test ' + intent_type)
            with open(intent_path + '/validate_' + intent_type + '.json') as json_file:
                test_json = json.load(json_file)
                test_iob, slot_types = nlu_benchmark_to_structured_iob(test_json, intent_type, nlp)
                test_slot_types.update(slot_types)
            train_samples += train_iob
            test_samples += test_iob
            intents.add(intent_type)

    train_slot_types = list(sorted(train_slot_types))
    test_slot_types = list(sorted(test_slot_types))
    intents = list(sorted(intents))

    train_set = {
        'data': train_samples,
        'meta': {
            'tokenizer': 'spacy',
            'language': 'en',
            'slot_types': train_slot_types,
            'intent_types': intents
        }
    }
    test_set = {
        'data': test_samples,
        'meta': {
            'tokenizer': 'spacy',
            'language': 'en',
            'slot_types': test_slot_types,
            'intent_types': intents
        }
    }

    if not os.path.exists('nlu-benchmark/preprocessed'):
        os.makedirs('nlu-benchmark/preprocessed')

    with open('nlu-benchmark/preprocessed/fold_train.json', 'w') as outfile:
        json.dump(train_set, outfile)

    with open('nlu-benchmark/preprocessed/fold_test.json', 'w') as outfile:
        json.dump(test_set, outfile)


def wit_preprocess(path, nlp):
    """Preprocess the wit dataset"""
    path_source = path + '/source'

    with open('{}/expressions.json'.format(path_source)) as json_file:
        expressions = json.load(json_file)

    samples, intent_types, slot_types = wit_to_structured_iob(expressions, nlp)

    dataset = np.array(samples)
    # do the stratified split on 5 folds, fixing the random seed
    slot_types = list(sorted(slot_types))
    intent_types = list(sorted(intent_types))
    # the value of intent for each sample, necessary to perform the stratified split (keeping distribution of intents in splits)
    intent_values = [s['intent'] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=dataset.size)
    folds_indexes = []
    for train_idx, test_idx in sss.split(np.zeros(len(intent_values)), intent_values):
        #print('train idx', train_idx, 'test idx', test_idx)
        folds_indexes.append(test_idx.tolist())

    print(folds_indexes)
    
    train, dev, final_test = (dataset[folds_indexes[0] + folds_indexes[1] + folds_indexes[2]], dataset[folds_indexes[3]],dataset[folds_indexes[4]])

    meta = {
        'tokenizer': 'spacy',
        'language': path[4:],
        'intent_types': intent_types,
        'slot_types': slot_types
    }

    if not os.path.exists('{}/preprocessed'.format(path)):
        os.makedirs('{}/preprocessed'.format(path))

    with open('{}/preprocessed/fold_train.json'.format(path), 'w') as outfile:
        json.dump({
            'data': train.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/fold_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': dev.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/final_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': final_test.tolist(),
            'meta': meta
        }, outfile)

def wit_to_structured_iob(expressions, nlp):
    #samples, entity_types = wit_to_structured_iob(expressions)
    slot_types = set()
    intent_types = set()
    items = expressions['data']
    samples = []
    for item in items:
        sentence = item['text']
        intent_type = None
        annots = []
        for e_or_i in item['entities']:
            if e_or_i['entity'] == 'intent':
                intent_type = e_or_i['value'].strip('"')
            else:
                entity_type = e_or_i['entity']
                if 'role' in e_or_i:
                    entity_type = e_or_i['role'] + '.' + entity_type
                annots.append((e_or_i['start'], e_or_i['end'], entity_type))

        words, slots = displacement_annotations_to_iob(sentence, annots, nlp)

        intent_types.add(intent_type)
        slot_types.update(slots)

        samples.append({
            'words': words,
            'length': len(words),
            'slots': slots,
            'intent': intent_type
        })

    return samples, intent_types, slot_types

def nlu_benchmark_to_structured_iob(data, intent_type, nlp):
    #train_iob, slot_types = nlu_benchmark_to_structured_iob(train_json, intent_type)
    iob_result = []
    slot_types = set()
    for sample in data[intent_type]:
        # build the annotations (start, end, slot_type)
        annots = []
        start_idx = 0
        sentence = ''
        words = []
        slots = []
        for span in sample['data']:
            slot = span.get('entity', None)
            if slot:
                annots.append((start_idx, start_idx + len(span['text']), slot))
            sentence += span['text']
            start_idx += len(span['text'])

        words, slots = displacement_annotations_to_iob(sentence, annots, nlp)
        slot_types.update(slots)

        iob_result.append({
            'words': words,
            'length': len(words),
            'slots': slots,
            'intent': intent_type
        })

    return iob_result, slot_types

def iob_lines_to_structured_iob(iob_lines):
    """
    Transforms an .iob file, whose lines are passed as parameters, to a structured representation.
    Example:
    BOS cheapest airfare from tacoma to orlando EOS	O B-cost_relative O O B-fromloc.city_name O B-toloc.city_name atis_airfare
    becomes
    {
        'tokenized': ['cheapest', 'airfare', 'from', 'tacoma', 'to', 'orlando'],
        'slots': ['B-cost_relative', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name'],
        'length': 6
        'intent': 'atis_airfare'
    }

    Each sample is put into a result object, together with an information about which tokenizer is used (on ATIS always space tokenizer):
    {
        'data': [LIST_OF_SAMPLES],
        'meta':{
            'tokenizer': 'spaces',
            'slot_types': [LIST_OF_FOUND_SLOT_VALUES]
            'intent_types': [LIST_OF FOUND_INTENT_VALUES]
        }
    }
    """

    slot_types = set()
    intent_types = set()
    data = []
    for line in iob_lines:
        # input is separated from outputs by a tab
        text, annotations = line.split('\t')
        # tokenization by space, removing BOS and EOS
        words = text.split()[1:-1]
        # also for the annotations, space-separated
        words_annotations = annotations.split()
        # slots annotations, removing the ones corresponding to BOS and EOS
        slots = words_annotations[1:-1]
        # the intent is the annotation corresponding to EOS
        intent = words_annotations[-1]

        assert len(words) == len(slots)
        length = len(words)

        # aggregated metadata
        slot_types.update(slots)
        intent_types.add(intent)
        
        data.append({
            'words': words,
            'slots': slots,
            'length': length,
            'intent': intent
        })

    slot_types = list(sorted(slot_types))
    intent_types = list(sorted(intent_types))

    return {
        'data': data,
        'meta': {
            'tokenizer': 'space',
            'language': 'en',
            'slot_types': slot_types,
            'intent_types': intent_types
        }
    }

def load_nlp(lang_name='en'):
    nlp = spacy.load(lang_name)
    return nlp

def displacement_annotations_to_iob(sentence, annotations, nlp):
    doc = nlp.make_doc(sentence)
    tags = biluo_tags_from_offsets(doc, annotations)

    words = []
    slots = []
    for word,tag in zip(doc,tags):
        tag = re.sub(r'^U', "B", tag)
        tag = re.sub(r'^L', "I", tag)
        #this occurs when multiple spaces exist
        word = word.text.strip()
        # tokenization makes some word  like " ", removing them
        if word:
            words.append(word)
            slots.append(tag)
    
    return words, slots

#atis_preprocess_old()
#wit_preprocess_old('wit_en')
#wit_preprocess_old('wit_it')

def main():
    nlp_en = load_nlp()
    nlp_it = load_nlp('it')
    which = os.environ.get('DATASET', None)
    print(which)
    if which is None:
        atis_preprocess()
        nlu_benchmark_preprocess(nlp_en)
        wit_preprocess('wit_en', nlp_en)
        wit_preprocess('wit_it', nlp_it)
    elif which == 'atis':
        atis_preprocess()
    elif which == 'nlu-benchnark':
        nlu_benchmark_preprocess(nlp_en)
    elif which == 'wit_en':
        wit_preprocess('wit_en', nlp_en)
    elif which == 'wit_it':
        wit_preprocess('wit_it', nlp_it)


if __name__ == '__main__':
    main()