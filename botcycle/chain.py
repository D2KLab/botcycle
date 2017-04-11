import nlu
import core
from pprint import pprint
import sentence_generation


class Chain:

    def __init__(self, tokens):
        self.units = {
            'nlu': nlu.Nlu(tokens),
            'core': core.Core(),
            'sentence_generation': sentence_generation.Sentence_generation()
        }
        self.utils = {"""
            'users_data': ,
            'contextual_data': ,
            'episodic_data':"""
        }

    def process(self, request):
        pprint(request)
        for k, unit in self.units.items():
            unit.process(request, self.utils)

        return request
