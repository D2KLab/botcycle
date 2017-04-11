from pprint import pprint
import nlu.wit_entities
import nlu.entity_resolver


class Nlu:
    def __init__(self, tokens):
        self.extractor = nlu.wit_entities.Extractor(tokens['wit.ai'])
        self.entity_resolver = nlu.entity_resolver.Resolver()

    def process(self, data, utils):
        data['nlu'] = {}
        if data['message']['type'] == 'text':
            intent, entities = self.extractor.parse(data['message']['text'])
            pprint(intent)
            pprint(entities)
            data['nlu']['intent'] = intent
            data['nlu']['entities'] = entities

            data['nlu']['entities_resolved'] = self.entity_resolver.resolve(data['chat_id'], entities)

        return data
