#!/bin/bash
# this is the script for the HQA2018 submission

set -e

pushd nlu
export DATASET=kvret
export OUTPUT_FOLDER=test
# TEST SET
# train and eval multi-turn approach
make train_joint
# train and eval single turn
FORCE_SINGLE_TURN=no_all make train_joint
# multi-turn without bot utterances
FORCE_SINGLE_TURN=no_bot_turn make train_joint
# multi-turn without previous intent
FORCE_SINGLE_TURN=no_previous_intent make train_joint
# multi-turn with CRF
#RECURRENT_MULTITURN=crf make train_joint
# LSTM
RECURRENT_MULTITURN=lstm make train_joint
# multi-turn without bot utterances
RECURRENT_MULTITURN=lstm FORCE_SINGLE_TURN=no_bot_turn make train_joint
# CRF
pushd joint
python train_crf.py
popd

# FINAL TEST SET
export MODE=finaltest
export OUTPUT_FOLDER=finaltest
# train and eval multi-turn approach
make train_joint
# train and eval single turn
FORCE_SINGLE_TURN=no_all make train_joint
# multi-turn without bot utterances
FORCE_SINGLE_TURN=no_bot_turn make train_joint
# multi-turn without previous intent
FORCE_SINGLE_TURN=no_previous_intent make train_joint
# multi-turn with CRF
#RECURRENT_MULTITURN=crf make train_joint
# LSTM
RECURRENT_MULTITURN=lstm make train_joint
# multi-turn without bot utterances
RECURRENT_MULTITURN=lstm FORCE_SINGLE_TURN=no_bot_turn make train_joint
# CRF
pushd joint
python train_crf.py finaltest
popd

popd