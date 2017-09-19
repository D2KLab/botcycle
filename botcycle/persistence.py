import os
import datetime
from pymongo import MongoClient


mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/botcycle')
client = MongoClient(mongodb_uri)

db = client.get_default_database()

users = db['users']

messages = db['messages']


def is_first_msg(chat_id):
    user = users.find_one({'_id': chat_id})
    if user:
        return False
    else:
        return True


def save_req(chat_id, message):
    time = datetime.datetime.utcnow()
    users.update_one({'_id': chat_id}, {"$set": {'time': time}}, upsert=True)
    messages.insert_one({'chat_id': chat_id, 'request': message, 'time': time})


def save_res(chat_id, response):
    time = datetime.datetime.utcnow()
    messages.insert_one(
        {'chat_id': chat_id, 'response': response, 'time': time})

def save_position(chat_id, position):
    position['time'] = datetime.datetime.utcnow()
    users.update_one({'_id': chat_id}, {"$set": {'last_position': position}})

def get_position(chat_id):
    user = users.find_one({'_id': chat_id})
    if not user:
        return None
    return user.get('last_position', None)

def save_user_token(chat_id, token):
    users.update_one({'_id': chat_id}, {"$set": {'access_token': token}})