import spacy
import sys

nlp = spacy.load('en', path='models')

print('Test your sentences.')
print('> ', end='', flush=True)

for line in sys.stdin:
    doc = nlp(line)
    for ent in doc.ents:
        print('entity: ', ent.label_, 'value:', ent.text)

    print('> ', end='', flush=True)