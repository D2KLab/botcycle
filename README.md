# BotCycle

BotCycle offers personalized recommendations of bike availabilities and contextual information of a city through a natural interaction via a chatbot.

## Modules

The module `brain` receives and sends messages and is the main component of the bot

The module `nlu` contains some experiments on Natural Language Understanding. For the moment it is standalone and not used by the `brain` module.

## Required libraries

This repository requires python 3

```bash
sudo apt install python3-pip python3-venv libgeos-dev
export LC_ALL=C # locale
sudo pip3 install virtualenv
```

## Setting up the environment

1. Clone/download this repository
2. For each of the modules install the dependencies in a virtual environment:
  - `virtualenv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

## Other requirements

The brain module requires some environment variables, see the module `README.md`

## Running the bot

Launch the bot: `python main.py`

## Using the bot

The bot understands three intents:

- search a bike
- search an empty slot
- plan a trip
- set the user position (useful for other intents with missing entities)

The position can be sent as an attachment in any moment, also when the bot does not make request.
