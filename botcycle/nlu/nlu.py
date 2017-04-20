from pprint import pprint
import nlu.wit_entities
import nlu.entity_resolver


class Nlu:
    def __init__(self, tokens):
        self.extractor = nlu.wit_entities.Extractor(tokens['wit.ai'])
        self.entity_resolver = nlu.entity_resolver.Resolver()

    def process(self, data, utils):
        data['nlu'] = {'entities': {}}
        if data['message']['type'] == 'text':
            intent, entities = self.extractor.parse(data['message']['text'])

            data['nlu']['intent'] = intent
            data['nlu']['entities'] = entities

            self.entity_resolver.resolve(data['chat_id'], entities, utils)

            # TODO get the last known user position (don't check validity of it there, it's on the core responsibility to check requirements)
            data['nlu']['entities']['user_position'] = None

        elif data['message']['type'] == 'location':
            # convert from location to place
            data['nlu']['entities']['user_position'] = self.entity_resolver.resolve_location(data['message']['location'], utils)
            data['nlu']['intent'] = {'value': 'set_position'}

        pprint(data['nlu']['entities'])

        return data
