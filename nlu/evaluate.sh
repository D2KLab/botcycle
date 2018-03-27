#!/bin/bash
# this is the script for the HQA2018 submission

set -e

# FINAL TEST SET
export OUTPUT_FOLDER=finaltest
export MODE=finaltest

#effects of different word embeddings
WORD_EMBEDDINGS=random DATASET=wit_it make train_joint
WORD_EMBEDDINGS=large DATASET=wit_it make train_joint

WORD_EMBEDDINGS=random DATASET=atis make train_joint
WORD_EMBEDDINGS=random DATASET=kvret make train_joint
WORD_EMBEDDINGS=random DATASET=nlu-benchmark make train_joint
WORD_EMBEDDINGS=random DATASET=wit_en make train_joint

WORD_EMBEDDINGS=small DATASET=atis make train_joint
WORD_EMBEDDINGS=small DATASET=kvret make train_joint
WORD_EMBEDDINGS=small DATASET=nlu-benchmark make train_joint
WORD_EMBEDDINGS=small DATASET=wit_en make train_joint

WORD_EMBEDDINGS=medium DATASET=atis make train_joint
WORD_EMBEDDINGS=medium DATASET=kvret make train_joint
WORD_EMBEDDINGS=medium DATASET=nlu-benchmark make train_joint
WORD_EMBEDDINGS=medium DATASET=wit_en make train_joint

WORD_EMBEDDINGS=large DATASET=atis make train_joint
WORD_EMBEDDINGS=large DATASET=kvret make train_joint
WORD_EMBEDDINGS=large DATASET=nlu-benchmark make train_joint
WORD_EMBEDDINGS=large DATASET=wit_en make train_joint

#effects of LSTM vs GRU
RECURRENT_CELL=gru DATASET=wit_it make train_joint
RECURRENT_CELL=gru DATASET=atis make train_joint
RECURRENT_CELL=gru DATASET=kvret make train_joint
RECURRENT_CELL=gru DATASET=nlu-benchmark make train_joint
RECURRENT_CELL=gru DATASET=wit_en make train_joint

# effect of attention or not (only on the slots!)
ATTENTION=none DATASET=wit_it make train_joint
ATTENTION=none DATASET=atis make train_joint
ATTENTION=none DATASET=kvret make train_joint
ATTENTION=none DATASET=nlu-benchmark make train_joint
ATTENTION=none DATASET=wit_en make train_joint