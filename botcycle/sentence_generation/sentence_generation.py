from pprint import pprint

class Sentence_generation:
    def __init__(self):
        pass

    def process(self, data):
        pprint(data)
        decision = data['decision']
        data['response'] = {}
        data['response']['type'] = 'text'
        data['response']['text'] = 'template response. ' + str(decision)
