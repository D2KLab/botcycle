import requests
import datetime
from .. import persistence

class WitWrapper:
    def __init__(self, token):
        self.token = token
        self.headers = {'Authorization':'Bearer {0}'.format(token)}

    def process(self, sentence):
        # with verbose queries, also returns start and end indexes of entities
        params = {'q':sentence, 'verbose': True, 'v': '20170920'}
        response = requests.get("https://api.wit.ai/message", params = params, headers = self.headers).json()

        response['time'] = datetime.datetime.utcnow()
        persistence.log_nlu(response)

        all_entities = response.get('entities', None)
        if all_entities == None:
            raise Exception('error with wit.ai')

        intent_list = all_entities.get('intent', None)
        if intent_list:
            intent = intent_list[0]

        else:
            intent = None

        entities = {}
        for key,value in all_entities.items():
            if key != 'intent':
                entities[key] = value[0]


        return intent, entities

def foo():
    print("hi")
