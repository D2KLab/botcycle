import os
import asyncio
import json
from queue import Queue
import traceback
import threading
import schedule
import time
import websockets
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, find_dotenv

# load environment from file if exists
load_dotenv(find_dotenv())

from botcycle import botcycle

outgoing_messages = Queue()


async def get_message(ws):
    message = await ws.recv()
    print(message)
    return json.loads(message)

def send_messages(websocket):
    """this is executed by a dedicated thread. Gets messages from the outgoing_messages queue and sends them on the websocket"""
    # set the event loop because this is another thread
    asyncio.set_event_loop(loop)
    try:
        while True:
            msg = outgoing_messages.get()
            print(msg)
            # run the async function in synchronous context
            future = asyncio.run_coroutine_threadsafe(websocket.send(json.dumps(msg)), loop)
            future.result()
    except websockets.exceptions.ConnectionClosed as e:
        # this exception occurred because the connection was closed
        # put back the message in the queue
        # TODO find the way to put on the first place, without using deque that does not block
        outgoing_messages.put(msg)


def queue_message(user_id, message, msg_type='text', buttons=None, markers=None):
    msg = {'userId': user_id, 'text': message,
           'type': msg_type, 'buttons': buttons, 'markers': markers}
    #print('enqueued a message')
    outgoing_messages.put(msg)


async def main():
    while True:
        try:
            async with websockets.connect(websocket_location) as websocket:
                print('connected to botkit')
                sender_thread = threading.Thread(target=send_messages, args=[websocket])
                sender_thread.daemon = True
                sender_thread.start()
                with ThreadPoolExecutor() as executor:
                    while True:
                        message = await get_message(websocket)
                        executor.submit(botcycle.process, message, queue_message)
                    

        except websockets.exceptions.ConnectionClosed as e:
            print(e)
        except OSError:
            print('unreachable botkit websocket')
        except Exception as e:
            traceback.print_exc()

        time.sleep(2)


websocket_path = os.environ.get('WEBSOCKET_PATH', 'main')
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

websocket_location = '{}://{}/{}?token={}'.format(
    ws_proto, botkit_location, websocket_path, websocket_token)

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

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
