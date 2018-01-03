"""
This module handles NLU requests.
Contains a wrapper for wit.ai and a wrapper for the neural network jointSLU
"""
import os
import datetime
from multiprocessing.pool import Pool

from .wit import WitWrapper
from .joint.inference import NeuralNetWrapper
from .. import persistence


def apply_async_wrapper(nlu, sentence):
    """
    A static wrapper for calling a class method in another process 
    """
    return nlu.process(sentence)


class Nlu(object):
    """
    Wrapper around both wit.ai and neural network jointSLU.
    Set the env variable NLU to 'both' (default), 'wit' or 'local' to switch
    """

    def __init__(self, token, language):
        # master switch between wit and local
        self.type = os.environ.get('NLU', 'both')
        if self.type == 'wit':
            self.real = WitWrapper(token)
        elif self.type == 'both':
            self.wit = WitWrapper(token)
            self.local = NeuralNetWrapper(language, 'wit_{}'.format(language))
            self.pool = Pool(processes=1)
        else:
            self.real = NeuralNetWrapper(language, 'wit_{}'.format(language))

    def process(self, sentence):
        """
        Turns a sentence into intent,entities
        """
        if self.type == 'both':
            # issue both, in separate threads to wait only max(t1,t2) instead of t1+t2
            async_result = self.pool.apply_async(apply_async_wrapper, (self.wit, sentence))
            nn_result = self.local.process(sentence)
            wit_result = async_result.get()
            # return only local processing, wit processing is done to keep the request on wit.ai
            result = nn_result
        else:
            result = self.real.process(sentence)

        intent, entities = result
        persistence.log_nlu({'_text': sentence, 'intent': intent, 'entities': entities, 'time': datetime.datetime.utcnow()})
        return result
