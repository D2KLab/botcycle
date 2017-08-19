import os
import asyncio
import json
import traceback
import websockets

from botcycle import botcycle


async def get_message(ws):
    message = await ws.recv()
    print(message)
    return json.loads(message)


def send_message(ws, user_id, message, attachments):
    msg_type = 'text'
    if attachments:
        msg_type = attachments[0]['type']
    if msg_type == 'button':
        msg_type = 'buttons'
    msg = {'userId': user_id, 'text': message, 'type': msg_type, 'attachments': attachments}
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
                        await botcycle.process(message, lambda chat_id, message, attachments=None: send_message(websocket, chat_id, message, attachments))
                    except Exception as e:
                        traceback.print_exc()

        except Exception as e:
            traceback.print_exc()
            print('trying to resume!')

websocket_token = os.environ.get(
    'WEBSOCKET_TOKEN', None) or 'pizza'  # TODO delete pizza
botkit_location = os.environ.get(
    'BOTKIT_LOCATION', None) or 'botcycle-botkit.herokuapp.com'
if not websocket_token:
    raise Exception('WEBSOCKET_TOKEN env variable missing!')

websocket_location = 'wss://{}/brain?jwt={}'.format(
    botkit_location, websocket_token)

asyncio.get_event_loop().run_until_complete(main())
