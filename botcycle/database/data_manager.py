# module main functions
# TODO provide access to different data sources (DB, pybikes, places)
from database.pybikes_wrapper import Pybikes_wrapper
from database.places_manager import Places_manager

class Data_manager:
    def __init__(self, tokens):
        self.bikeshare = {}
        self.bikeshare['torino'] = Pybikes_wrapper('to-bike')
        self.places_manager = Places_manager(tokens)
