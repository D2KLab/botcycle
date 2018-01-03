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
        Turns a sentence into intent+entities
        """
        if self.type == 'both':
            # issue both, in separate threads to wait only max(t1,t2) instead
            # of t1+t2
            async_result = self.pool.apply_async(self.wit.process, (sentence))
            nn_result = self.local.process(sentence)
            wit_result = async_result.get()
            # return only local processing
            result = nn_result
        else:
            result = self.real.process(sentence)

        result['time'] = datetime.datetime.utcnow()
        persistence.log_nlu(result)
        return result
