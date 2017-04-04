import os
import sys
import time
import json
import requests
from pprint import pprint
import telepot
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
import spacy
import pybikes
import witEntities

def on_chat_message(msg):
    content_type, chat_type, chat_id =telepot.glance(msg)
    user_data_path = 'users_data/' + str(chat_id)
    if not os.path.isdir(user_data_path):
        os.makedirs(user_data_path)
        bot.sendMessage(chat_id, "Welcome! I am BotCycle and can give you bike sharing informations")

    #print(content_type, chat_type, chat_id)
    results = stations_with_bikes
    if content_type == 'text':
        log_msg(chat_id, msg['text'])
        intent, entities = extractor.parse(msg['text'])
        log_entities(chat_id, intent, entities)

        if intent:
            if intent['value'] == 'search_bike':
                #bot.sendMessage(chat_id, "You want to search a bike")
                search_bike(chat_id, entities)

            elif intent['value'] == 'search_slot':
                #bot.sendMessage(chat_id, "You want to search an empty slot")
                search_slot(chat_id, entities)

            elif intent['value'] == 'plan_trip':
                #bot.sendMessage(chat_id, "You want to plan a trip")
                plan_trip(chat_id, entities)

            elif intent['value'] == 'set_position':
                #bot.sendMessage(chat_id, "You want to set the position")
                set_position_str(chat_id, entities)

            elif intent['value'] == 'ask_position':
                askPosition(chat_id)

            elif intent['value'] == 'greeting':
                response = 'Hi there! How can I help?'
                log_response(chat_id, response)
                bot.sendMessage(chat_id, response)

            elif intent['value'] == 'thank':
                response = 'You\'re welcome!'
                log_response(chat_id, response)
                bot.sendMessage(chat_id, response)

            elif intent['value'] == 'info':
                response = 'I am BotCycle, a bot that can give you informations about bike sharing in your city.\nTry to ask me something! ;)'
                log_response(chat_id, response)
                bot.sendMessage(chat_id, response)

            else:
                bot.sendMessage(chat_id, "Unexpected intent: " + intent['value'])

        else:
            bot.sendMessage(chat_id, "Your sentence does not have an intent")

    elif content_type == 'location':
        set_position(chat_id, msg['location'])
    else:
        bot.sendMessage(chat_id, "why did you send " + content_type + "?")


# working on global variables?? SRSLY?
def update_data():
    torino_bikeshare.update()
    torino_stations = {x.name:x for x in torino_bikeshare.stations}
    stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
    stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]

def search_nearest(position, results_set):
    distance_sq = float('inf')
    best = -1
    #print("results_set has size: " + str(len(results_set)))
    for idx, val in enumerate(results_set):
        d2 = (position['latitude']-val.latitude) **2 + (position['longitude']-val.longitude) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = idx

    return results_set[best]

def set_position_str(chat_id, entities):
    location = getLocation(chat_id, entities)
    #print(location)
    if location:
        set_position(chat_id, location)
        bot.sendLocation(chat_id, location['latitude'], location['longitude'])

def set_position(chat_id, location):
    location['time'] = time.strftime("%c")
    with open('users_data/'+str(chat_id)+'/last_position', 'w+') as last_position_file:
        json.dump(location, last_position_file)

    response = "Ok I got your position"
    log_response(chat_id, response)
    bot.sendMessage(chat_id, response)

def askPosition(chat_id):
    markup = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Send position', request_location=True)]])
    response = 'Where are you? Use the button below or just tell me!'
    log_response(chat_id, response)
    bot.sendMessage(chat_id, response, reply_markup=markup)

def provideResult(chat_id, station, search_type):
    if search_type == 'bikes':
        response = "You can find " + str(station.bikes) + " free bikes at station " + station.name

    elif search_type == 'slots':
        response = "You can find " + str(station.free) + " empty slots at station " + station.name

    log_response(chat_id, response)
    bot.sendMessage(chat_id, response)
    bot.sendLocation(chat_id, station.latitude, station.longitude)

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
            bot.sendMessage(chat_id, response)

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

    result = search_nearest(location, stations_with_bikes)
    provideResult(chat_id, result, 'bikes')

def search_slot(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        return

    result = search_nearest(location, stations_with_free)
    provideResult(chat_id, result, 'slots')

def plan_trip(chat_id, entities):
    location = getLocation(chat_id, entities)
    loc_from_str = getEntity(entities, 'from')
    loc_to_str = getEntity(entities, 'to')

    loc_from = loc_to = None

    if loc_from_str:
        loc_from = search_place(loc_from_str)
        if not loc_from:
            response = 'I could not find a place named ' + loc_from_str
            log_response(chat_id, response)
            bot.sendMessage(chat_id, response)

    if loc_to_str:
        loc_to = search_place(loc_to_str)
        if not loc_to:
            response = 'I could not find a place named ' + loc_to_str
            log_response(chat_id, 'I could not find a place named ' + loc_to_str)
            bot.sendMessage(chat_id, response)

    if not loc_from and not loc_to:
        response = "Your trip has no origin and no destination"
        log_response(chat_id, response)
        bot.sendMessage(chat_id, response)
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

    result_from = search_nearest(loc_from, stations_with_bikes)
    result_to = search_nearest(loc_to, stations_with_free)

    provideResult(chat_id, result_from, 'bikes')
    provideResult(chat_id, result_to, 'slots')

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



# load the token from file
with open(sys.argv[1]) as tokens_file:
    data = json.load(tokens_file)
    telegram_token = data['telegram']
    wit_token = data['wit.ai']

# TODO enable this fro nlp stuff. Now only dealing with fixed queries
#nlp = spacy.load('en')

extractor = witEntities.Extractor(wit_token)

torino_bikeshare = pybikes.get('to-bike')
torino_bikeshare.update()
torino_stations = {x.name:x for x in torino_bikeshare.stations}
stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]

bot = telepot.Bot(telegram_token)
pprint(bot.getMe())
bot.message_loop({'chat': on_chat_message})

while 1:
    # keep updating the bike-sharing data every 1 min
    time.sleep(60)
    torino_bikeshare.update()
    torino_stations = {x.name:x for x in torino_bikeshare.stations}
    stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
    stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]
