import os
import time
import requests

from . import bikes
from .nlu import wit
from . import persistence
from . import personalization

sendMessageFunction = None


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
            chat_id, 'Thanks for your contribution! Every person that logs in helps me provide better results to everyone!')
        return

    content_type = 'text' if (msg['text'] != '') else (
        'location' if (msg.get('position', None) != None) else 'other')

    if persistence.is_first_msg(chat_id):
        sendMessageFunction(
            chat_id, "Welcome! I am BotCycle and can give you bike sharing informations")

    #print(content_type, chat_type, chat_id)
    if content_type == 'text':
        intent, entities = extractor.parse(msg['text'])

        if msg['text'] == '/start':
            sendMessageFunction(
                chat_id, "I am BotCycle. Try to ask me something about bike sharing!")
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
                chat_id, 'please login with facebook to improve recommendations', 'login')
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
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'thank':
                response = 'You\'re welcome!'
                sendMessageFunction(chat_id, response)

            elif intent['value'] == 'info':
                response = 'I am BotCycle, a bot that can give you informations about bike sharing in your city.\nTry to ask me something! ;)'
                sendMessageFunction(chat_id, response)

            else:
                sendMessageFunction(
                    chat_id, "Unexpected intent: " + intent['value'])

        else:
            sendMessageFunction(
                chat_id, "Your sentence does not have an intent")

    elif content_type == 'location':
        set_position(chat_id, msg['position'])
    else:
        sendMessageFunction(chat_id, "why did you send " + content_type + "?")


def set_position_str(chat_id, entities):
    location = getLocation(chat_id, entities)
    # print(location)
    if location:
        set_position(chat_id, location, verbose=False)
        markers = [{'type': 'location', 'latitude': location['latitude'],
                    'longitude': location['longitude']}]
        sendMessageFunction(chat_id, 'Ok I got your position',
                            msg_type='map', markers=markers)


def set_position(chat_id, location, verbose=True):
    global sendMessageFunction
    persistence.save_position(chat_id, location)
    if verbose:
        response = "Ok I got your position"
        sendMessageFunction(chat_id, response)


def askPosition(chat_id):
    global sendMessageFunction
    response = 'Where are you?'
    sendMessageFunction(chat_id, response, msg_type='request_location')


def provideResult(chat_id, station, search_type, buttons=None):
    global sendMessageFunction
    if not station:
        response = "Impossible to find informations"

    elif search_type == 'bikes':
        response = "You can find " + \
            str(station.bikes) + " free bikes at station " + station.name

    elif search_type == 'slots':
        response = "You can find " + \
            str(station.free) + " empty slots at station " + station.name

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
        print("error in communication with nominatim.openstreetmap.org: " + e)

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
            response = 'I could not find a place named ' + location_name
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
    sendMessageFunction(chat_id, 'You could try this interesting place: ' +
                        rec['name'], msg_type='map', markers=markers, buttons=buttons)


def search_bike(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
        return

    city, result = bikes.search_nearest(location, 'bikes')
    provideResult(chat_id, result, 'bikes', buttons=askFeedback())

    recommend(chat_id, [result])


def search_slot(chat_id, entities):
    location = getLocation(chat_id, entities)
    if not location:
        askPosition(chat_id)
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
            response = 'I could not find a place named ' + loc_from_str
            sendMessageFunction(chat_id, response)

    if loc_to_str:
        loc_to = search_place(loc_to_str)
        if not loc_to:
            response = 'I could not find a place named ' + loc_to_str
            sendMessageFunction(chat_id, response)

    if not loc_from and not loc_to:
        response = "Your trip has no origin and no destination"
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

    city1, result_from = bikes.search_nearest(loc_from, 'bikes')
    city2, result_to = bikes.search_nearest(loc_to, 'slots')

    if city1 and city2 and city1 is not city2:
        response = 'Your trip starts at ' + city1 + ' and ends at ' + city2 + \
            '. You cannot take a bike from one city and go to another one!'
        sendMessageFunction(chat_id, response)
        return

    provideResult(chat_id, result_from, 'bikes')
    provideResult(chat_id, result_to, 'slots', buttons=askFeedback())

    recommend(chat_id, [result_from, result_to])


def askFeedback():
    return [{'type': 'text', 'value': '👍'}, {'type': 'text', 'value': '👎'}]

wit_token = os.environ['WIT_TOKEN']
extractor = wit.Extractor(wit_token)
