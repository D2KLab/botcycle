SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

download:
	pushd data && bash download.sh && popd

preprocess:
	pushd data && python preprocess.py && popd

train_atis_entities:
	DATASET=atis MAX_ITERATIONS=300 python -m spacy_entities.train

train_wit_entities:
	DATASET=wit MAX_ITERATIONS=300 python -m spacy_entities.train

train_atis_intents:
	DATASET=atis MODEL_NAME=bidirectional_gru MAX_ITERATIONS=50 python -m intent_classifier.model_rnn

train_wit_intents:
	DATASET=wit MODEL_NAME=bidirectional_lstm MAX_ITERATIONS=100 python -m intent_classifier.model_rnn

test_wit:
	INTENT_MODEL_PATH=`exec ls -d1 intent_classifier/models/wit/** | sort -n | tail -1` DATASET=wit LANG_MODEL_PATH=spacy_entities/models/wit python -m intent_classifier.test_intent