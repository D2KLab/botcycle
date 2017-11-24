# Nlu
Experiments with Natural Language Understanding

To run the examples, the requirements need to be installed and the available tasks are defined in the makefile.

## Data

This folder contains the datasets:

- wit dataset. To download simply run `make download` script (you need `WIT_TOKEN` env variable specific for your wit.ai application, which can be found under Settings in the Wit console)
- atis dataset

## Spacy_entities

Contains code that trains the NER of spacy on the wit_data. Run the `train.py` script that will load the `'en'` model and after training will save the updated model in the subfolder `model`.

## Intent classifier

Contains some models for classifying the intent of the sentences. As above, the training data comes from the wit.ai export.

## Joint slot-filling and intent classification

The folder `joint` contains a tensorflow implementation of a neural network where the two tasks are done together.

Use `make train_joint` to run it.
