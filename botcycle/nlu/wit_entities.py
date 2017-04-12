import requests

class Extractor:
    def __init__(self, token):
        self.token = token
        self.headers = {'Authorization':'Bearer {0}'.format(token)}

    def parse(self, sentence):
        params = {'q':sentence}
        response = requests.get("https://api.wit.ai/message", params = params, headers = self.headers).json()
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
                if key == 'location' or key == 'to' or key == 'from':
                    entities[key]['type'] = 'location'

        return intent, entities

def foo():
    print("hi")
