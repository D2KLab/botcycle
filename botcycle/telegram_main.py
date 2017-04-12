import os
import sys
import json
import telepot
import time
from pprint import pprint
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from chain import Chain
import urllib3
import telepot.api

# proxy stuff https://github.com/nickoala/telepot/issues/83
myproxy_url = os.environ.get('HTTPS_PROXY')
if myproxy_url:
    telepot.api._pools = {'default': urllib3.ProxyManager(proxy_url=myproxy_url, num_pools=3, maxsize=10, retries=False, timeout=30),}
    telepot.api._onetime_pool_spec = (urllib3.ProxyManager, dict(proxy_url=myproxy_url, num_pools=1, maxsize=1, retries=False, timeout=30))


def on_chat_message(msg):

    # TODO put the message in standard form, ready for the chain
    request = normalizeRequest(msg)

    pprint(request)

    # TODO process the request in the chain
    results = chain.process(request)

    for result in results:
        # TODO provide the result
        sendResponse(result)


# this function provide uniform message structure across multiple apps
# (telegram,messenger)
def normalizeRequest(msg):
    result = {}
    content_type, chat_type, chat_id = telepot.glance(msg)
    result['chat_id'] = chat_id
    result['message'] = {}
    result['message']['type'] = content_type
    result['message']['time'] = time.time()
    if content_type == 'text':
        result['message']['text'] = msg['text']

    elif content_type == 'location':
        result['message']['location'] = msg['location']

    return result


def sendResponse(data):
    response = data['response']
    if response['type'] == 'text':
        bot.sendMessage(data['chat_id'], response['text'])

    elif response['type'] == 'location_request':
        markup = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text='Send position', request_location=True)]])
        response = response['text']
        bot.sendMessage(chat_id, response, reply_markup=markup)


# load the token from file
with open(sys.argv[1]) as tokens_file:
    tokens_data = json.load(tokens_file)
    telegram_token = tokens_data['telegram']

# TODO create chain
chain = Chain(tokens_data)

bot = telepot.Bot(telegram_token)
pprint(bot.getMe())
bot.message_loop({'chat': on_chat_message})

while 1:
    time.sleep(60)

    # TODO update data
