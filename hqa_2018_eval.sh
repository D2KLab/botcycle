#!/bin/bash
# this is the script for the HQA2018 submission

set -e

pushd nlu
export DATASET=kvret

# train and eval multi-turn approach
make train_joint
# train and eval single turn
FORCE_SINGLE_TURN=no_all make train_joint
# multi-turn without bot utterances
FORCE_SINGLE_TURN=no_bot_turn make train_joint
# multi-turn without previous intent
FORCE_SINGLE_TURN=no_previous_intent make train_joint
# multi-turn with CRF
RECURRENT_MULTITURN=crf make train_joint

popd