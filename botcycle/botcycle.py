import sys
import time
import json
import requests
from pprint import pprint
import telepot
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
import spacy
import pybikes

def on_chat_message(msg):
    content_type, chat_type, chat_id =telepot.glance(msg)
    print(content_type, chat_type, chat_id)
    results = stations_with_bikes
    if content_type == 'text':
        #doc = nlp(msg['text'])

        if msg['text'][0] == '/':
            if msg['text'][1] == 'b':
                results = stations_with_bikes

            elif msg['text'][1] == 'f':
                results = stations_with_free

            elif msg['text'][1] == 'p':
                #simpified: user asks by name of station
                station = torino_stations.get(msg['text'], None);
                if station:
                    response = "station " + msg['text'] + ":\nbikes:" + str(station.bikes) + "\nfree:" + str(station.free)
                    bot.sendMessage(chat_id, response)
                else:
                    # ask for location
                    places_found = requests.get('http://nominatim.openstreetmap.org/search?format=json&q=' + msg['text'][3:]).json()
                    if len(places_found) > 0:
                        user_positions[chat_id] = {}
                        user_positions[chat_id]['latitude'] = float(places_found[0]['lat'])
                        user_positions[chat_id]['longitude'] = float(places_found[0]['lon'])
                        bot.sendMessage(chat_id, "Got your position: " + places_found[0]['display_name'])
                    else:
                        bot.sendMessage(chat_id, "Unable to get this place")

            else:
                # other '/' command
                bot.sendMessage(chat_id, "i don't understand")

            if user_positions.get(chat_id, None) == None:
                markup = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Send position', request_location=True)]])
                bot.sendMessage(chat_id, 'Where are you?', reply_markup=markup)

            else:
                res = search_nearest(user_positions[chat_id], results)
                bot.sendMessage(chat_id, res.name + ":\nbikes:" + str(res.bikes) + "\nfree:" + str(res.free))

        else:
            markup = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Send position', request_location=True)]])
            bot.sendMessage(chat_id, 'Where are you?', reply_markup=markup)

    elif content_type == 'location':
        user_positions[chat_id] = msg['location']
        bot.sendMessage(chat_id, "Ok I got your position: " + str(user_positions[chat_id]['latitude']) + ";" + str(user_positions[chat_id]['longitude']))
    else:
        bot.sendMessage(chat_id, "why did you send " + content_type + "?")

# working on global variables?? SRSLY?
def update_data():
    torino_bikeshare.update()
    torino_stations = {x.name:x for x in torino_bikeshare.stations}
    stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
    stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]

def search_nearest(user_position, results_set):
    distance_sq = float('inf')
    best = -1
    print("results_set has size: " + str(len(results_set)))
    for idx, val in enumerate(results_set):
        d2 = (user_position['latitude']-val.latitude) **2 + (user_position['longitude']-val.longitude) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = idx

    return results_set[best]

# load the token from file
with open(sys.argv[1]) as tokens_file:
    data = json.load(tokens_file)
    telegram_token = data['telegram']

# TODO enable this fro nlp stuff. Now only dealing with fixed queries
#nlp = spacy.load('en')

torino_bikeshare = pybikes.get('to-bike')
torino_bikeshare.update()
torino_stations = {x.name:x for x in torino_bikeshare.stations}
stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]

# TODO persistency
user_positions = {}

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
