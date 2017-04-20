# module main functions
# TODO provide access to different data sources (DB, pybikes, places)
import sqlite3
import os.path
from database.pybikes_wrapper import Pybikes_wrapper
from database.places_manager import Places_manager
import database.db_initializer

class Data_manager:
    def __init__(self, tokens):
        self.bikeshare = {}
        self.bikeshare['torino'] = Pybikes_wrapper('to-bike')
        if not os.path.isfile('data.db'):
            database.db_initializer.execute_init_sql('data.db')
        self.db_conn = sqlite3.connect('data.db', check_same_thread = False)
        self.places_manager = Places_manager(tokens, self.db_conn)
