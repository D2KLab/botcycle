import os
import spacy
import sys

MY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ['DATASET']
MODEL_PATH = os.environ.get('MODEL_PATH', '{}/models/{}'.format(MY_PATH, DATASET))

nlp = spacy.load('en', path=MODEL_PATH)

print('Test your sentences.')
print('> ', end='', flush=True)

for line in sys.stdin:
    doc = nlp(line)
    for ent in doc.ents:
        print('entity: ', ent.label_, 'value:', ent.text)

    print('> ', end='', flush=True)