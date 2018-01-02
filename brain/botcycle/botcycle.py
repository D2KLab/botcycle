import os
import time
import requests

from math import radians, cos, sin, asin, sqrt

from . import bikes
from .nlu import wit
from . import persistence
from . import personalization
from . import output_sentences

LANGUAGE = os.environ.get('BOT_LANGUAGE', 'EN')
print('language is ' + LANGUAGE)

sendMessageFunction = None

# chat contexts
contexts = {}


def process(msg, sendMessage):
    global sendMessageFunction
    sendMessageFunction = sendMessage

    chat_id = msg['userId']

    # TODO improve type checking
    msg_type = msg.get('type', None)
    if msg_type == 'login':
        print('user ' + chat_id + ' logged in; received code')
        personalization.add_data_from_login(
            chat_id, msg['facebookId'], msg['token'])
        sendMessageFunction(
            chat_id, output_sentences.get(LANGUAGE, 'THANK_LOGGED_IN'))
        return

    content_type = 'text' if (msg['text'] != '') else (
        'location' if (msg.get('position', None) != None) else 'other')

    if persistence.is_first_msg(chat_id):
        sendMessageFunction(
            chat_id, output_sentences.get(LANGUAGE, 'FIRST_MESSAGE'))

    #print(content_type, chat_type, chat_id)
    if content_type == 'text':
        intent, entities = extractor.process(msg['text'])

        if msg['text'] == '/start':
            sendMessageFunction(
                chat_id, output_sentences.get(LANGUAGE, 'FIRST_MESSAGE'))
            return

        if msg['text'] == '👍':
            # TODO collect positive feedback
            return

        if msg['text'] == '👎':
            # TODO collect negative feedback
            return

        # TODO this is to test facebook login
        if msg['text'] == 'login':
            sendMessageFunction(
                chat_id, output_sentences.get(LANGUAGE, 'ASK_LOGIN') , 'login')
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
                context_continue(chat_id)

            elif intent['value'] == 'ask_position':
                askPosition(chat_id)

            elif intent['value'] == 'greeting':
                response = output_sentences.get(LANGUAGE, 'GREET_BACK')
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'thank':
                response = output_sentences.get(LANGUAGE, 'THANK_BACK')
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'info':
                response = output_sentences.get(LANGUAGE, 'PROVIDE_INFO')
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'city_supported':
                get_nearest_supported_city(chat_id, entities)

            elif intent['value'] == 'booking':
                response = output_sentences.get(LANGUAGE, 'CANT_BOOK')
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'end_discussion':
                persistence.save_end_of_sequence(chat_id)

            else:
                sendMessageFunction(
                    chat_id, output_sentences.get(LANGUAGE, 'UNEXPECTED_INTENT').format(intent=intent['value']))

        else:
            sendMessageFunction(
                chat_id, output_sentences.get(LANGUAGE, 'NO_INTENT'))

    elif content_type == 'location':
        set_position(chat_id, msg['position'])
        context_continue(chat_id)
    else:
        sendMessageFunction(chat_id, output_sentences.get(LANGUAGE, 'UNSUPPORTED_CONTENT_TYPE').format(type=content_type))


def set_position_str(chat_id, entities):
    location = getLocation(chat_id, entities)
    if location is None:
        sendMessageFunction(chat_id, output_sentences.get(LANGUAGE, 'REQUIRED_POSITION'))
    # print(location)
    if location:
        set_position(chat_id, location, verbose=False)
        markers = [{'type': 'location', 'latitude': location['latitude'],
                    'longitude': location['longitude']}]
        sendMessageFunction(chat_id, output_sentences.get(LANGUAGE, 'ACK_POSITION'),
                            msg_type='map', markers=markers)


def set_position(chat_id, location, verbose=True):
    global sendMessageFunction
    print('saving current position')
    persistence.save_position(chat_id, location)
    if verbose:
        response = output_sentences.get(LANGUAGE, 'ACK_POSITION')
        sendMessageFunction(chat_id, response)


def askPosition(chat_id):
    global sendMessageFunction
    response = output_sentences.get(LANGUAGE, 'ASK_POSITION')
    sendMessageFunction(chat_id, response, msg_type='request_location')


def provideResult(chat_id, station, search_type, buttons=None):
    global sendMessageFunction
    if not station:
        response = output_sentences.get(LANGUAGE, 'ERROR_SEARCHING')

    elif search_type == 'bikes':
        response = output_sentences.get(LANGUAGE, 'FREE_BIKES').format(count=station.bikes, station_name=station.name)

    elif search_type == 'slots':
        response = output_sentences.get(LANGUAGE, 'FREE_SLOTS').format(count=station.bikes, station_name=station.name)

    if station:
        markers = [{'type': 'location', 'latitude': station.latitude,
                    'longitude': station.longitude}]
        sendMessageFunction(chat_id, response, msg_type='map',
                            markers=markers, buttons=buttons)
    else:
        sendMessageFunction(chat_id, response, buttons=buttons)


def search_place(place_name):
    result = {}
    try:
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?key=' +
                                os.environ['MAPS_TOKEN'] + '&address=' + place_name).json()
    except Exception as e:
        print(output_sentences.get(LANGUAGE, 'GEOCODING_ERROR').format(searched=place_name))

    places_found = response['results']

    if len(places_found) > 0:
        result['latitude'] = float(
            places_found[0]['geometry']['location']['lat'])
        result['longitude'] = float(
            places_found[0]['geometry']['location']['lng'])

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

    user_position = persistence.get_position(chat_id)

    if location_obj:
        location_name = location_obj.get('value', None)
        location = search_place(location_name)
        if not location:
            response = output_sentences.get(LANGUAGE, 'GEOCODING_ERROR').format(searched=location_name)
            sendMessageFunction(chat_id, response)

    elif user_position:
        # TODO check when it was set
        location = user_position

    else:
        location = None
    return location


def recommend(chat_id, results):
    return
    # TODO this should be done in separate thread
    # TODO this is very rude
    global sendMessageFunction
    recs = personalization.get_recommend_places(chat_id, results)
    rec = recs[0]
    time.sleep(3)

    markers = [{'type': 'location', 'latitude': rec['location']
                ['lat'], 'longitude': rec['location']['lng']}]
    # TODO on messenger and skype no more than three buttons per card
    buttons = askFeedback()
    buttons.append({'type': 'link', 'title': 'details', 'value': rec['url']})
    sendMessageFunction(chat_id, output_sentences.get(LANGUAGE, 'RECOMMEND_PLACE').format(place_name=rec['name']), msg_type='map', markers=markers, buttons=buttons)


def search_bike(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        save_context(chat_id, 'search_bike', entities)
        return

    city, result = bikes.search_nearest(location, 'bikes')
    provideResult(chat_id, result, 'bikes', buttons=askFeedback())

    return
    recommend(chat_id, [result])

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def get_nearest_supported_city(chat_id, entities):
    global sendMessageFunction
    location = getLocation(chat_id, entities)

    tag, meta = bikes.nearest_city_find(location)
    supported_city_name = meta['city']
    result_latlng = search_place(supported_city_name)
    # is the city if distance < 500m
    if (haversine(location['longitude'], location['latitude'], result_latlng['longitude'], result_latlng['latitude']) < 0.5):
        response = output_sentences.get(LANGUAGE, 'SUPPORTED_AFFIRMATIVE').format(city=meta['city'])
    else:
        response = output_sentences.get(LANGUAGE, 'SUPPORTED_NEGATIVE').format(nearest_city=meta['city'])
    sendMessageFunction(chat_id, response)


def search_slot(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        save_context(chat_id, 'search_bike', entities)
        return

    city, result = bikes.search_nearest(location, 'slots')
    provideResult(chat_id, result, 'slots', buttons=askFeedback())

    recommend(chat_id, [result])


def plan_trip(chat_id, entities):
    global sendMessageFunction
    location = getLocation(chat_id, entities)
    loc_from_str = getEntity(entities, 'from')
    loc_to_str = getEntity(entities, 'to')

    loc_from = loc_to = None

    if loc_from_str:
        loc_from = search_place(loc_from_str)
        if not loc_from:
            response = output_sentences.get(LANGUAGE, 'GEOCODING_ERROR').format(searched=loc_from_str)
            sendMessageFunction(chat_id, response)

    if loc_to_str:
        loc_to = search_place(loc_to_str)
        if not loc_to:
            response = output_sentences.get(LANGUAGE, 'GEOCODING_ERROR').format(searched=loc_to_str)
            sendMessageFunction(chat_id, response)

    if not loc_from and not loc_to:
        response = output_sentences.get(LANGUAGE, 'INCOMPLETE_TRIP')
        sendMessageFunction(chat_id, response)
        return

    if not loc_from or not loc_to:
        # if only one of them is missing, ca use the user location as backup
        if not location:
            askPosition(chat_id)
            save_context(chat_id, 'search_bike', entities)
            return

        else:
            if loc_from:
                loc_to = location

            else:
                loc_from = location

    city1, result_from = bikes.search_nearest(loc_from, 'bikes')
    city2, result_to = bikes.search_nearest(loc_to, 'slots')

    if city1 and city2 and city1 is not city2:
        response = output_sentences.get(LANGUAGE, 'INTERCITY_TRIP').format(source=city1, destination=city2)
        sendMessageFunction(chat_id, response)
        return

    provideResult(chat_id, result_from, 'bikes')
    provideResult(chat_id, result_to, 'slots', buttons=askFeedback())

    recommend(chat_id, [result_from, result_to])

def save_context(chat_id, intent, entities):
    contexts[chat_id] = {'intent': intent, 'entities': entities}

def context_continue(chat_id):
    print('context_continue for user' + chat_id)
    context = contexts.pop(chat_id, None)
    print(context)
    if context:
        if context['intent'] == 'search_bike':
            search_bike(chat_id, context['entities'])
        elif context['intent'] == 'search_slot':
            search_slot(chat_id, context['entities'])
        elif context['intent'] == 'plan_trip':
            plan_trip(chat_id, context['entities'])


def askFeedback():
    return [{'type': 'text', 'value': '👍'}, {'type': 'text', 'value': '👎'}]

wit_token = os.environ['WIT_TOKEN_' + LANGUAGE]
extractor = wit.Extractor(wit_token, LANGUAGE)
