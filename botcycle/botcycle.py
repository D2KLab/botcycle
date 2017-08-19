import os
import sys
import time
import json
import requests
from pprint import pprint
import asyncio
#from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
import spacy
import pybikes
from botcycle.witEntities import witEntities
import urllib3
import telepot.api

sendMessageFunction = None

async def process(msg, sendMessage):
    global sendMessageFunction
    sendMessageFunction = sendMessage

    content_type =  'text' if (msg['text'] != '') else ('location' if (msg.get('position', None) != None) else 'other')
    chat_id = msg['userId']
    user_data_path = 'users_data/' + str(chat_id)
    if not os.path.isdir(user_data_path):
        os.makedirs(user_data_path)
        sendMessageFunction(chat_id, "Welcome! I am BotCycle and can give you bike sharing informations")

    #print(content_type, chat_type, chat_id)
    if content_type == 'text':
        log_msg(chat_id, msg['text'])
        intent, entities = extractor.parse(msg['text'])
        log_entities(chat_id, intent, entities)

        if msg['text'] == '/start':
            sendMessageFunction(chat_id, "I am BotCycle. Try to ask me something about bike sharing!")
            return

        if msg['text'] == '👍':
            #TODO collect positive feedback
            return

        if msg['text'] == '👎':
            #TODO collect negative feedback
            return


        if intent:
            if intent['value'] == 'search_bike':
                #sendMessageFunction(chat_id, "You want to search a bike")
                search_bike(chat_id, entities)

            elif intent['value'] == 'search_slot':
                #sendMessageFunction(chat_id, "You want to search an empty slot")
                search_slot(chat_id, entities)

            elif intent['value'] == 'plan_trip':
                #sendMessageFunction(chat_id, "You want to plan a trip")
                plan_trip(chat_id, entities)

            elif intent['value'] == 'set_position':
                #sendMessageFunction(chat_id, "You want to set the position")
                set_position_str(chat_id, entities)

            elif intent['value'] == 'ask_position':
                askPosition(chat_id)

            elif intent['value'] == 'greeting':
                response = 'Hi there! How can I help?'
                log_response(chat_id, response)
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'thank':
                response = 'You\'re welcome!'
                log_response(chat_id, response)
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'info':
                response = 'I am BotCycle, a bot that can give you informations about bike sharing in your city.\nTry to ask me something! ;)'
                log_response(chat_id, response)
                sendMessageFunction(chat_id, response)

            else:
                sendMessageFunction(chat_id, "Unexpected intent: " + intent['value'])

        else:
            sendMessageFunction(chat_id, "Your sentence does not have an intent")

    elif content_type == 'location':
        set_position(chat_id, msg['location'])
    else:
        sendMessageFunction(chat_id, "why did you send " + content_type + "?")


# working on global variables?? SRSLY?
def update_data(which_to_update):
    print('update_data called on : ' + str(which_to_update))
    result = {}
    for value in which_to_update:
        print('going to get info on ' + value)
        try:
            bikeshare = pybikes.get(value);
            bikeshare.update()

            result[value] = {
                'city': bikeshare.meta['city'],
                'stations': {x.name:x for x in bikeshare.stations},
                'with_bikes': [station for station in bikeshare.stations if station.bikes>0],
                'with_slots': [station for station in bikeshare.stations if station.free>0]
            }

        except Exception as e:
            print('something bad while getting info for ' + value + ': ' + str(e) + '\n, discarding this city')
            which_to_update.remove(value)



    return result

    """
    torino_bikeshare = pybikes.get('to-bike')
    torino_bikeshare.update()
    torino_stations = {x.name:x for x in torino_bikeshare.stations}
    stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
    stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]"""

def get_city_cached(position):
    global bike_info, to_update

    tag = nearest_city_find(position)
    result = bike_info.get(tag, None)
    if not result:
        to_update.append(tag)
        bike_info = update_data(to_update)
        # now the city must be there or something bad happened
        result = bike_info.get(tag, None)

    return result


def search_nearest(position, search_type):

    info = get_city_cached(position)

    if not info:
        return None, None

    if search_type == 'bikes':
        results_set = info['with_bikes']

    else:
        results_set = info['with_slots']

    distance_sq = float('inf')
    best = -1
    #print("results_set has size: " + str(len(results_set)))
    for idx, val in enumerate(results_set):
        d2 = (position['latitude']-val.latitude) **2 + (position['longitude']-val.longitude) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = idx

    if best is -1:
        return info['city'], None

    return info['city'], results_set[best]

def set_position_str(chat_id, entities):
    location = getLocation(chat_id, entities)
    #print(location)
    if location:
        set_position(chat_id, location)
        attachments = [{'type': 'location', 'latitude': location['latitude'], 'longitude': location['longitude']}]
        sendMessageFunction(chat_id, '', attachments=attachments)

def set_position(chat_id, location):
    global sendMessageFunction
    location['time'] = time.strftime("%c")
    with open('users_data/'+str(chat_id)+'/last_position', 'w+') as last_position_file:
        json.dump(location, last_position_file)

    response = "Ok I got your position"
    log_response(chat_id, response)
    sendMessageFunction(chat_id, response)

def askPosition(chat_id):
    global sendMessageFunction
    #markup = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Send position', request_location=True)]], resize_keyboard=True, one_time_keyboard=True)
    attachments = [{'type': 'button', 'value': 'Send position'}]
    response = 'Where are you? Use the button below or just tell me!'
    log_response(chat_id, response)
    sendMessageFunction(chat_id, response, attachments=attachments)

def provideResult(chat_id, station, search_type, attachments=None):
    global sendMessageFunction
    if not station:
        response = "Impossible to find informations"

    elif search_type == 'bikes':
        response = "You can find " + str(station.bikes) + " free bikes at station " + station.name

    elif search_type == 'slots':
        response = "You can find " + str(station.free) + " empty slots at station " + station.name

    log_response(chat_id, response)
    sendMessageFunction(chat_id, response, attachments=attachments)
    if station:
        attachments = [{'type': 'location', 'latitude': station.latitude, 'longitude': station.longitude}]
        sendMessageFunction(chat_id, '', attachments=attachments)

def search_place(place_name):
    result = {}
    try:
        places_found = requests.get('http://nominatim.openstreetmap.org/search?format=json&q=' + place_name).json()
    except Exception as e:
        print("error in communication with nominatim.openstreetmap.org: " + e)

    if len(places_found) > 0:
        result['latitude'] = float(places_found[0]['lat'])
        result['longitude'] = float(places_found[0]['lon'])

    return result

def getEntity(entities, key):
    entity_obj = entities.get(key, None)
    if entity_obj:
        result = entity_obj.get('value', None)

    else:
        result = None

    return result

def getLocation(chat_id, entities):
    global sendMessageFunction
    # TODO use getEntity
    location_obj = entities.get('location', None)
    user_position = None
    try:
        with open('users_data/'+str(chat_id)+'/last_position', 'r') as last_position_file:
            user_position = json.load(last_position_file)
    except FileNotFoundError:
        pass

    if location_obj:
        location_name = location_obj.get('value', None)
        location = search_place(location_name)
        if not location:
            response = 'I could not find a place named ' + location_name
            sendMessageFunction(chat_id, response)

    elif user_position:
        # TODO check when it was set
        location = user_position

    else:
        location = None
    return location

def search_bike(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        return

    city, result = search_nearest(location, 'bikes')
    provideResult(chat_id, result, 'bikes', attachments=askFeedback())


def search_slot(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        return

    city, result = search_nearest(location, 'slots')
    provideResult(chat_id, result, 'slots', attachments=askFeedback())


def plan_trip(chat_id, entities):
    global sendMessageFunction
    location = getLocation(chat_id, entities)
    loc_from_str = getEntity(entities, 'from')
    loc_to_str = getEntity(entities, 'to')

    loc_from = loc_to = None

    if loc_from_str:
        loc_from = search_place(loc_from_str)
        if not loc_from:
            response = 'I could not find a place named ' + loc_from_str
            log_response(chat_id, response)
            sendMessageFunction(chat_id, response)

    if loc_to_str:
        loc_to = search_place(loc_to_str)
        if not loc_to:
            response = 'I could not find a place named ' + loc_to_str
            log_response(chat_id, 'I could not find a place named ' + loc_to_str)
            sendMessageFunction(chat_id, response)

    if not loc_from and not loc_to:
        response = "Your trip has no origin and no destination"
        log_response(chat_id, response)
        sendMessageFunction(chat_id, response)
        return

    if not loc_from or not loc_to:
        # if only one of them is missing, ca use the user location as backup
        if not location:
            askPosition(chat_id)
            return

        else:
            if loc_from:
                loc_to = location

            else:
                loc_from = location

    city1, result_from = search_nearest(loc_from, 'bikes')
    city2, result_to = search_nearest(loc_to, 'slots')

    if city1 and city2 and city1 is not city2:
        response = 'Your trip starts at ' + city1 + ' and ends at ' + city2 + '. You cannot take a bike from one city to another one!'
        log_response(chat_id, response)
        sendMessageFunction(chat_id, response)
        return

    provideResult(chat_id, result_from, 'bikes')
    provideResult(chat_id, result_to, 'slots', attachments=askFeedback())


def askFeedback():
    #return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='👍'), KeyboardButton(text='👎')]], resize_keyboard=True, one_time_keyboard=True)
    return [{'type': 'button', 'value': '👍'}, {'type': 'button', 'value': '👎'}]


def log_msg(chat_id, msg):
    log(chat_id, '-->' + msg)

def log_entities(chat_id, intent, entities):
    log(chat_id, 'intent:' + str(intent) + ' entities: ' + str(entities))

def log_response(chat_id, response):
    log(chat_id, '<--' + response)

def log(chat_id, string):
    # 1 means line buffered
    with open('users_data/' + str(chat_id) + '/chat.log', 'a', 1) as log_file:
        log_file.write(time.strftime("%c") + "\t" + string + '\n')


def nearest_city_find(position):
    where_to_search = {}
    for schema in pybikes.get_all_data():
        data = pybikes.get_data(schema)
        instances = data.get('instances', None)
        if not instances:
            instances = []
            for key, value in data['class'].items():
                instances.extend(value['instances'])

        for instance in instances:
            where_to_search[instance['tag']] = instance['meta']

    # now where_to_search contains a map (tag, {latitude, longitude, ...})
    #pprint(where_to_search)

    distance_sq = float('inf')
    best = -1
    #print("results_set has size: " + str(len(results_set)))
    for key, value in where_to_search.items():
        d2 = (position['latitude']-value['latitude']) **2 + (position['longitude']-value['longitude']) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = key

    #print("nearest is " + best)
    return best


wit_token = os.environ['WIT_TOKEN']

# TODO enable this fro nlp stuff. Now only dealing with fixed queries
#nlp = spacy.load('en')

extractor = witEntities.Extractor(wit_token)

to_update = []
bike_info = update_data(to_update)

async def keepRunningUpdates():
    while 1:
        # keep updating the bike-sharing data every 1 min
        try:
            time.sleep(60)
            bike_info = update_data(to_update)

        except Exception as e:
            print('something bad happened: ' + str(e))


# TODO re-enable periodic updates
#asyncio.ensure_future(keepRunningUpdates())