SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

download:
	pushd data && bash download.sh && popd

preprocess:
	pushd data && python preprocess.py && popd

train_atis_entities:
	DATASET=atis MAX_ITERATIONS=4 python -m spacy_entities.train

train_wit_entities:
	DATASET=wit MAX_ITERATIONS=50 python -m spacy_entities.train

train_atis_intents:
	echo TODO

train_wit_intents:
	echo TODO

test_wit_entities:
	DATASET=wit python -m spacy_entities.test

test_wit_intents:
	echo TODO