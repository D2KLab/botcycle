# personalized-chatbot

## Setting up the environment

1. Clone/download this repository
2. Install the dependencies:
  - `virtualenv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

This repository for now requires python 3

## Get required tokens

### Telegram tokens

In order to authenticate the bot on Telegram, it is required to ask to [BotFather](https://telegram.me/BotFather) a token and place it into a file named `tokens.json` as in the following:

```json
{
  "telegram": "PUT HERE YOUR TELEGRAM TOKEN"
}
```

## Running the bot

Launch the bot: `python botcycle/botcycle.py tokens.json`

## Using the bot

The bot simply understands three commands:

- `/p <search a place>`: to set the position:
  - if the search string matches exactly a name of a station, the position of the station is set
  - otherwise asks to openstreetmap for a translation from string to position, and sets the user position to the result
  - if also the request to openstreetmap fails (no results), the position is not set
- `/b` asks for the nearest station to the user position, having bikes available
- `/f` asks for the nearest station to the user position, having at least one free slot

The position can be sent as an attachment in any moment, also when the bot does not make request.
