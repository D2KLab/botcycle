# Data

This folder contains different datasets. For each one there is the source version and the preprocessed one (where the differences in data annotation have been removed).

## ATIS

Charles T Hemphill, John J Godfrey, and George R Doddington. 1990. "The ATIS spoken language systems pilot corpus". _In Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27, 1990_.

The dataset ATIS contains intent and slots annotations for single-turned dialogs. The corpus has been retrieved from https://github.com/yvchen/JointSLU/tree/master/data.
Since the split is done with spaces, and the default tokenization with SpaCy is a bit different, this dataset needs the tokenizer `space` to align the word embeddings retrieval inside the tensorflow `py_func`.

## KVRET

Mihail Eric and Christopher D Manning. 2017. "Key-Value Retrieval Networks for Task-Oriented Dialogue". _In Proceedings of the 18th Annual SIGdial MeetingDiscourse and Dialogue_. Association for Computational Linguistics, Saarbrücken, Germany, 37–49.

The dataset KVRET contains intent and slots annotations for multi-turn dialogs. The corpus has been retrieved from https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/.

## Nlu-benchmark

The dataset contains intent and slots annotations for single-turn dialogs. The corpus has been retrieved from https://github.com/snipsco/nlu-benchmark.

## Wit_en and wit_it

Those datasets have been collected using https://wit.ai platform. Intent and slots annotations for single-turn dialogs.
For updating this dataset the following steps are required:

- have a valid wit.ai token and put it into the `.env` file in the `data` folder
- run `make download` to get the archive, and extract the useful files
- run `DATASET=wit_xx make preprocess`


## Multiturn_en and multiturn_it

The collection of those datasets exploits the dumps of the operational database, that extracted to the folder `exported` is then appended to the file `multiturn_xx/source/tabular.tsv` for being able to edit it in a simple way, and thanks to the file `multiturn_xx/source/stats.tsv` only the new messages are analyzed.
The result is a multi-turn annotation of intents and slots.

To update the contents with the last messages logged in the operational database run the following:

- run `make export_messages` to export the mongoDB collections `messages.json` and `nlu_history.json` into the folder `exported/xx/`
- run `make extract_tsv` to append the new messages to the `multiturn_xx/source/tabular.tsv` file and update the last extraction time in the file `multiturn_xx/source/stats.json`
- manually review the annotations on the new lines of the file `multiturn_xx/source/tabular.tsv`
- run `DATASET=multiturn_xx make preprocess` to save the sessions in a usable format and do the train/test/finaltest split in the folder `multiturn_xx/preprocessed/`
