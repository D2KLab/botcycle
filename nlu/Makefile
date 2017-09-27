SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

download:
	pushd data && bash download.sh && popd

preprocess:
	pushd data && python preprocess.py && popd

train_entities:
	DATASET=wit MAX_ITERATIONS=1000 python -m spacy_entities.train

train_intents:
	echo TODO