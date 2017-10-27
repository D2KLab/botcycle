# This script downloads a pretrained word embedding and prepares it for SpaCy

pushd glove_wiki_it
# download the pretrained glove
wget http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz -o pretrained.tar.gz
# extract the tarball
tar -xzvf pretrained.tar.gz
# move files in current directory
mv home/berardi/* ./ && rm -r home
# now export to single text file that contains: string, vector
python - <<EOF
from gensim.models.word2vec import Word2Vec
model = Word2Vec.load('glove_WIKI')
model.wv.save_word2vec_format('pretrained.txt') 
EOF
# remove the first line of the file
tail -n +2 pretrained.txt > pretrained.new && mv -f pretrained.new pretrained.txt
# compress the txt as bz2
bzip2 pretrained.txt
# create a SpaCy compatible binary file
python - <<EOF
import spacy
spacy.vocab.write_binary_vectors('pretrained.txt.bz2','pretrained.bin')
EOF

popd
# now simply load with nlp.vocab.load_vectors_from_bin_loc('path/pretrained.bin')