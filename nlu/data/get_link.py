from __future__ import print_function
import sys
import requests

# the token is passed as command line argument
wit_token = sys.argv[1]
wit_app_name = sys.argv[2]
res = requests.get('https://api.wit.ai/inst/MartinoMensio/{}'.format(wit_app_name), headers = {'Authorization':'Bearer {0}'.format(wit_token)}).json()

print(res.get('private_handle'))