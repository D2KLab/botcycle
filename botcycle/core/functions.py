"""
This file contains the functions that implement the steps required for the processing of an intent (defined in the brain.json file).
The functions are wrappers around the different kinds of operations that the core module orchestrates, and have a common interface
that is used by the core to be able to call them without need to know what they do. They always have as a parameter the data that
contains all the processing informations (from chat text to all the values added by the pipeline) and another parameter that is
passed as the brain.json file specifies.
"""

def find_bike(data, utils, args):
    # TODO get position from args (must be passed by the core)
    result = utils['data_manager'].bikeshare['torino'].search_nearest({'latitude':44, 'longitude': 7}, feature='bike')
    data['info'] = {}
    data['info']['stations'] = [result]

def find_slot(data, utils, args):
    # TODO get position from args (must be passed by the core)
    result = utils['data_manager'].bikeshare['torino'].search_nearest({'latitude':44, 'longitude': 7}, feature='slot')
    data['info'] = {}
    data['info']['stations'] = [result]

def check_meteo(data, utils, args):
    pass

def set_position(data, utils, args):
	pass