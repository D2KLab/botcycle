SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

download:
	pushd data && bash download.sh && popd

create_it_embeddings:
	pushd data/embeddings && bash get_glove_wiki_it.sh && popd

preprocess:
	pushd data && python preprocess.py && popd

export_messages:
	pushd data && bash export.sh && popd

extract_tsv:
	pushd data && python extract_tsv.py en && python extract_tsv.py it && popd

train_atis_entities:
	DATASET=atis MAX_ITERATIONS=300 python -m spacy_entities.train

train_wit_entities:
	DATASET=wit_en MAX_ITERATIONS=300 python -m spacy_entities.train

train_atis_intents:
	DATASET=atis MODEL_NAME=bidirectional_gru MAX_ITERATIONS=50 python -m intent_classifier.model_rnn

train_wit_intents:
	DATASET=wit_en SPACY_MODEL_NAME=en MODEL_NAME=bidirectional_gru MAX_ITERATIONS=100 python -m intent_classifier.model_rnn

test_wit:
	INTENT_MODEL_PATH=`exec ls -d1 intent_classifier/models/wit_en/** | sort -n | tail -1` DATASET=wit LANG_MODEL_PATH=spacy_entities/models/wit python -m intent_classifier.test_intent

train_joint:
	python -m joint.main

test_joint:
	pushd ../brain/botcycle/nlu/ && python -m joint.inference && popd

build_models:
	DATASET=wit_en $(MAKE) train_joint && DATASET=wit_it $(MAKE) train_joint &&\
	 mkdir -p ../brain/botcycle/nlu/joint/results/ && cp -r joint/results/last/wit_* ../brain/botcycle/nlu/joint/results/