# This script downloads a pretrained word embedding and prepares it for SpaCy

pushd glove_wiki_it
echo "downloading the pretrained vectors"
# download the pretrained glove
wget -v --show-progress http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz -o pretrained.tar.gz
# extract the tarball
tar -xzvf pretrained.tar.gz
# move files in current directory
mv home/berardi/* ./ && rm -r home
# now export to single text file that contains: string, vector
python - <<EOF
from gensim.models.word2vec import Word2Vec
print('loading the glove files')
model = Word2Vec.load('glove_WIKI')
print('exporting to textual word2vec format')
model.wv.save_word2vec_format('pretrained.txt') 
EOF

echo "preparing spacy vectors"

python - <<EOF
import numpy
import spacy
from spacy.language import Language

nlp = spacy.blank('it')

def update_progress(curr, tot):
    workdone = curr/tot
    print("\rProgress: [{0:50s}] {1:.1f}% - {2}/{3}".format('#' * int(workdone * 50), workdone*100, curr, tot), end="", flush=True)

with open('pretrained.txt', 'rb') as file_:
    header = file_.readline()
    nr_row, nr_dim = header.split()
    nr_row = int(nr_row)
    nlp.vocab.reset_vectors(width=int(nr_dim))
    for idx, line in enumerate(file_):
        line = line.rstrip().decode('utf8')
        pieces = line.rsplit(' ', int(nr_dim))
        word = pieces[0]
        vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
        nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
        if idx % 500 == 0:
          update_progress(idx, nr_row)

# now save the vectors
nlp.vocab.vectors.to_disk('glove_wiki_it/spacy_vectors_it')
EOF

popd
# now simply load with nlp.vocab.load_vectors_from_bin_loc('path/pretrained.bin')