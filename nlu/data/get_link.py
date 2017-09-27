from __future__ import print_function
from dotenv import load_dotenv, find_dotenv
import os
import requests


# load environment from file if exists
load_dotenv(find_dotenv())

wit_token = os.environ['WIT_TOKEN']
res = requests.get('https://api.wit.ai/inst/MartinoMensio/BotCycle', headers = {'Authorization':'Bearer {0}'.format(wit_token)}).json()

print(res.get('private_handle'))