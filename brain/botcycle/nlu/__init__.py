import os
from .wit import WitWrapper
from .joint.inference import NeuralNetWrapper
    

class Nlu(object):

    def __init__(self, token, language):
        # master switch between wit and local
        self.type = os.environ.get('NLU', 'local')
        if self.type == 'wit':
            self.real = WitWrapper(token)
        elif self.type == 'both':
            self.wit = WitWrapper(token)
            self.local = NeuralNetWrapper(language, 'wit_{}'.format(language))
        else:
            self.real = NeuralNetWrapper(language, 'wit_{}'.format(language))

    def process(self, sentence):
        if self.type == 'both':
            #issue both 
            wit_result = self.wit.process(sentence)
            nn_result = self.local.process(sentence)
            # return only local processing
            return wit_result
        else:
            return self.real.process(sentence)
