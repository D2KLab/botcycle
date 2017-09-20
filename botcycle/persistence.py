import os
import datetime
from pymongo import MongoClient


mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/botcycle')
client = MongoClient(mongodb_uri)

db = client.get_default_database()

users = db['users']

messages = db['messages']

facebook_users = db['facebook_users']


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

def save_facebook_token(chat_id, facebook_id, access_token):
    """save the link between chat_id and facebook_id, and store the token for the facebook_user"""
    users.update_one({'_id': chat_id}, {"$set": {'facebook_id': facebook_id}})
    facebook_users.update_one({'_id': facebook_id}, {"$set": {'access_token': access_token}}, upsert=True)

def get_facebook_user(facebook_id):
    return facebook_users.find_one({'_id': facebook_id})

def get_facebook_users():
    return facebook_users.find({})