import json
import os

# get the absolute path to the data folder, so that no problem of where
# this is called
DATA_PATH = os.path.dirname(os.path.abspath(__file__))


def load_data(dataset_name):
    """Returns the data_splitted, entity_types, intent_types.
    
    data_splitted can be [fold1, fold2, fold3, fold4, fold5] or [test, train], look at the size"""
    path = DATA_PATH + '/' + dataset_name + '/preprocessed'
    with open(path + '/entity_types.json') as json_file:
        entity_types = json.load(json_file)

    with open(path + '/intent_types.json') as json_file:
        intent_types = json.load(json_file)

    fold_files = os.listdir(path)
    fold_files = sorted([f for f in fold_files if f.startswith('fold_')])

    data_splitted = []
    for file_name in fold_files:
        with open(path + '/' + file_name) as json_file:
            data_splitted.append(json.load(json_file))

    return data_splitted, entity_types, intent_types

# print(load_data('wit'))
# print(load_data('atis'))
