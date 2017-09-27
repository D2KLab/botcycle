"""
This module preprocesses the datasets in equivalent formats
"""
import json
import os
import numpy as np


def atis_preprocess():
    """
    preprocesses the atis dataset, taking as source the files atis.test.w-intent.iob and
    atis.train.w-intent.iob

    Produces in output the files entity_types.json, fold_train.json, fold_test.json
    """
    # TODO define flag to use entity.subentity or only entity
    with open('atis/source/atis.test.w-intent.iob') as txt_file:
        raw_test_set = txt_file.read()
    test_set = atis_normalize_raw_set(raw_test_set)
    with open('atis/source/atis.train.w-intent.iob') as txt_file:
        train_set = txt_file.readlines()
    train_tagged, entity_types, intent_types = atis_lines_to_json(train_set)
    test_tagged, test_entity_types, test_intent_types = atis_lines_to_json(test_set)

    # some entities and intents may appear only in test
    entity_types.update(test_entity_types)
    intent_types.update(test_intent_types)
    entity_types = list(sorted(entity_types))
    intent_types = list(sorted(intent_types))

    with open('atis/intent_types.json', 'w') as outfile:
        json.dump(intent_types, outfile)

    with open('atis/entity_types.json', 'w') as outfile:
        json.dump(entity_types, outfile)

    with open('atis/fold_train.json', 'w') as outfile:
        json.dump(train_tagged, outfile)

    with open('atis/fold_test.json', 'w') as outfile:
        json.dump(test_tagged, outfile)


def atis_normalize_raw_set(content):
    """Put the test set in the same form as the train set"""
    items = content.split('\n\n')
    results = []
    for item in items:
        lines = item.splitlines()
        chunks = np.array((len(lines), 2))
        chunks = np.array(list(map(lambda l: np.array(l.split()), lines)))
        if chunks.size:
            line = '{}\t{}'.format(
                ' '.join(chunks[:, 0]), ' '.join(chunks[:, 1]))
            results.append(line)

    return results


def atis_lines_to_json(content):
    """Transforms the content (list of lines) in json,
    detecting entity start and end indexes in sentences.
    Returns the tagged dataset, the enitity_types and the intent_types"""

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
        # TODO handle multi-intent
        intent_types.add(intent)
        element['intent'] = intent
        # chunks are defined by the space, IOB notations correspond to this
        # split
        chunks = text[:start_annotations_idx - 1].split()
        entities = []
        state = 'O'
        entity = {}
        for idx, tag in enumerate(entities_tags):
            tag = tag.split('-')
            next_state = tag[0]
            if len(tag) == 2:
                if next_state == 'B':
                    if state == 'B':
                        # close previous entity
                        entity['end'] = sum(map(len, chunks[:idx])) + idx - 1
                        entity['value'] = element['text'][entity['start']:entity['end']]
                        entities.append(entity)
                    # beginning of new entity
                    entity = {'type': tag[1], 'start': sum(
                        map(len, chunks[:idx])) + idx}
                    entity_types.add(tag[1])

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


def wit_preprocess():
    """Preprocesses the wit.ai dataset from the folder wit/BotCycle.
    To download the updated dataset, use the download.sh script.
    Saves the tagged dataset, the enitity_types and the intent_types"""
    enitites_path = 'wit/BotCycle/entities'
    entities_files = os.listdir(enitites_path)
    # remove .json suffix
    entity_types = list(map(lambda file_name: file_name[:-5], entities_files))
    entity_types.remove('intent')

    with open(enitites_path + '/intent.json') as json_file:
        intents = json.load(json_file)
    
    intent_types = list(map(lambda val: val['value'], intents['data']['values']))

    with open('wit/BotCycle/expressions.json') as json_file:
        expressions = json.load(json_file)

    dataset = wit_get_normalized_data(expressions)

    # perform the split on 5 folds
    dataset = np.array(dataset)
    # initialize the random generator seed to the size of the dataset, just to
    # make it split always the same
    np.random.seed(dataset.size)
    np.random.shuffle(dataset)
    fold_size = len(dataset) // 5
    folds = [dataset[:fold_size], dataset[fold_size:2 * fold_size],
             dataset[2 * fold_size:3 * fold_size], dataset[3 * fold_size:4 * fold_size], dataset[4 * fold_size:]]

    for idx, fold in enumerate(folds):
        with open('wit/fold_{}.json'.format(idx + 1), 'w') as outfile:
            json.dump(fold.tolist(), outfile)

    with open('wit/intent_types.json', 'w') as outfile:
        json.dump(intent_types, outfile)

    with open('wit/entity_types.json', 'w') as outfile:
        json.dump(entity_types, outfile)


def wit_get_normalized_data(expressions):
    """Returns a list of objects like `{'text': SENTENCE, 'intent': ,
    'entities': [{'entity': ENTITY_NAME, 'value': ENTITY_VALUE, 'start': INT, 'end', INT}]}`"""
    # TODO consider role ??
    items = expressions['data']
    results = []
    for item in items:
        result = {'text': item['text'], 'intent': None, 'entities': []}
        for e_or_i in item['entities']:
            if e_or_i['entity'] == 'intent':
                result['intent'] = e_or_i['value'].strip('"')
            else:
                entity = {'type': e_or_i['entity'], 'value': e_or_i['value'].strip(
                    '"'), 'start': e_or_i['start'], 'end': e_or_i['end']}
                result['entities'].append(entity)

        results.append(result)

    return results


atis_preprocess()
wit_preprocess()
