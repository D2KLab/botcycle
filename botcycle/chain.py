import nlu
import core
import database
from pprint import pprint
import sentence_generation


class Chain:

    def __init__(self, tokens):
        self.nlu = nlu.Nlu(tokens)
        self.core = core.Core()
        self.sentence_generation = sentence_generation.Sentence_generation()
        self.units = [
            self.nlu,
            self.core,
            self.sentence_generation
        ]
        self.utils = {
            'data_manager': database.Data_manager(tokens)
        }

    def process(self, request):
        pprint(request)
        for unit in self.units:
            unit.process(request, self.utils)

        core_next = request.get('core_next', None)
        # if the response has more output, need to call again the core to pop the previous talk
        if core_next and core_next['enabled']:
            result = self.process(core_next)
            result.append(request)
            return result.reverse()

        return [request,]
