import os
import asyncio
import json
import traceback
import threading
import schedule
import time
import websockets
from dotenv import load_dotenv, find_dotenv

# load environment from file if exists
load_dotenv(find_dotenv())

from botcycle import botcycle


async def get_message(ws):
    message = await ws.recv()
    print(message)
    return json.loads(message)


def send_message(ws, user_id, message, msg_type, buttons, markers):
    msg = {'userId': user_id, 'text': message,
           'type': msg_type, 'buttons': buttons, 'markers': markers}
    print('sending a message')
    print(msg)
    asyncio.ensure_future(ws.send(json.dumps(msg)))


async def main():
    while True:
        try:
            async with websockets.connect(websocket_location) as websocket:
                print('connected to botkit')
                while True:
                    message = await get_message(websocket)
                    try:
                        await botcycle.process(message, lambda chat_id, message, msg_type='text', buttons=None, markers=None: send_message(websocket, chat_id, message, msg_type, buttons, markers))
                    except Exception as e:
                        traceback.print_exc()

        except Exception as e:
            traceback.print_exc()
            print('trying to resume!')
            time.sleep(1)

websocket_token = os.environ.get(
    'WEBSOCKET_TOKEN', None)
# the websocket token is compulsory
if not websocket_token:
    raise Exception('WEBSOCKET_TOKEN env variable missing!')
# default location on heroku
botkit_location = os.environ.get(
    'BOTKIT_LOCATION', 'botcycle-botkit.herokuapp.com')
# default using secured web socket, unless environment variable changes
ws_proto = os.environ.get('WS_PROTO', 'wss')

websocket_location = '{}://{}/brain?jwt={}'.format(
    ws_proto, botkit_location, websocket_token)

"""
The job_thread will execute this function, that every 5 seconds checks
if some jobs need execution
"""


def job_monitor():
    while 1:
        #print('checking jobs')
        schedule.run_pending()
        time.sleep(5)


job_thread = threading.Thread(target=job_monitor)
job_thread.daemon = True
job_thread.start()

asyncio.get_event_loop().run_until_complete(main())
