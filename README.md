# BotCycle

BotCycle offers personalized recommendations of bike availabilities and contextual information of a city through a natural interaction via a chatbot.

## Modules

The module `brain` receives and sends messages and is the main component of the bot

The module `nlu` contains some experiments on Natural Language Understanding. For the moment it is standalone and not used by the `brain` module.

## Installation without docker-compose

### Required libraries

This repository requires python 3

```bash
sudo apt install python3-pip python3-venv libgeos-dev
export LC_ALL=C # locale
sudo pip3 install virtualenv
```

### Setting up the environment

1. Clone/download this repository
2. For each of the modules install the dependencies in a virtual environment:
  - `virtualenv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

### Training the language models

To use the language models for NLU at runtime, it is required to train them.

The training is done on the preprocessed datasets `wit_en` and `wit_it` that can be found in the `nlu/data` folder.

The training and saving can be done by running `cd nlu && make build_models`, that will place in the `brain/botcycle/nlu/joint/results/` folder the results that will be loaded later at runtime.

### Running the bot

Launch the bot: `python main.py`

## Installation with docker-compose

Alternatively, you can use docker-compose

### Requirements

The instruction refer to ubuntu distribution. You can adapt them easily to other linux distributions.

1. Install `docker-compose` package: `sudo apt install docker-compose`
2. Check the status of the docker-daemon: `systemctl status docker`
  1. If the daemon is not running, start it: `systemctl start docker`. Consider using auto-start of the daemon at system boot (`systemctl enable docker` / `systemctl disable docker`)
3. Allow your user to run docker without sudo:
  1. Create the docker group if not exist: `sudo groupadd docker`
  2. Add the desired user to the docker group: `sudo gpasswd -A $USER docker`
  3. Either do a `newgrp docker` or log out/in to activate the changes to groups

### Training the language models

When using the dockerized components, training the models can be a bit more complicated.

If you have the dependencies installed on your host (directly or indirectly via `virtualenv`), you can follow the instructions relative to "without docker" choice, and thanks to volume mapping they will be available inside the containers.

Instead if you want to train the models from the containers the steps are the following:

- start one of the bot containers (`docker start CONTAINER_NAME` or using `docker-compose start`)
- enter interactively in it: `docker exec -it CONTAINER_NAME bash`
- go to the makefile location: `cd /nlu`
- run the correct make target: `make build_models`
- exit from the container

### Running the bot

1. Make sure you are in top folder of this repository, where the `docker-compose.yml` file is located
2. Run `docker-compose up` to:
  1. Build the images (if not yet available)
  2. Create the containers (if not created yet)
  3. Create the network (if not created yet)
  4. Start the containers

After the first run, you will be in interactive mode: the console output of the containers are routed to your console. To stop the containers type CTRL+C. For successive runs, you can simply use `docker-compose start` that will start the containers and keep them in the background. You can stop then by running `docker-compose stop`, or see the logs of a container with `docker-compose logs CONTAINER_NAME`.

## Other requirements

The brain module requires some environment variables, see the module `README.md`.

You may need the mongo cli to explore the database: `sudo apt install mongo-clients`

## Using the bot

The bot understands three intents:

- search a bike
- search an empty slot
- plan a trip
- set the user position (useful for other intents with missing entities)

The position can be sent as an attachment in any moment, also when the bot does not make request.
