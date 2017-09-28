SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

download:
	pushd data && bash download.sh && popd

preprocess:
	pushd data && python preprocess.py && popd

train_atis_entities:
	DATASET=atis MAX_ITERATIONS=500 python -m spacy_entities.train

train_wit_entities:
	DATASET=wit MAX_ITERATIONS=500 python -m spacy_entities.train

train_atis_intents:
	DATASET=atis MODEL_NAME=mean_1l MAX_ITERATIONS=500 python -m intent_classifier.mean_model

train_wit_intents:
	DATASET=wit MODEL_NAME=mean_1l MAX_ITERATIONS=500 python -m intent_classifier.mean_model

test_wit_entities:
	DATASET=wit python -m spacy_entities.test

test_wit_intents:
	echo TODO