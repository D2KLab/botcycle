
class Sentence_generation:
    def __init__(self):
        pass

    def process(self, data, utils):
        decision = data['decision']
        data['response'] = {}
        data['response']['type'] = 'text'
        data['response']['text'] = 'template response. ' + decision
