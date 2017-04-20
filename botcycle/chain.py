import nlu
import core
import database
from pprint import pprint
import sentence_generation


class Chain:

    def __init__(self, tokens, response_sender):
        self.nlu = nlu.Nlu(tokens)
        self.core = core.Core()
        self.sentence_generation = sentence_generation.Sentence_generation()
        self.response_sender = response_sender
        self.units = [
            self.nlu,
            self.core
        ]
        self.utils = {
            'generate_response': self.generate_response,
            'data_manager': database.Data_manager(tokens)
        }

    def process(self, request):
        pprint(request)
        for unit in self.units:
            unit.process(request, self.utils)

        #self.generate_response(request) #already done in the core

    def generate_response(self, data):
        self.sentence_generation.process(data)
        self.response_sender(data)
