import tensorflow as tf
import numpy as np

class EmbeddingsFromScratch(object):
  
    def __init__(self, vocab, name, embedding_size):
        """
        vocab is a list of words
        """
        vocab = list(vocab) + ['<UNK>']
        self.vocab_size = len(vocab)
        vocab_tensor = tf.constant(vocab)
        self.word2index = tf.contrib.lookup.index_table_from_tensor(vocab_tensor, default_value=self.vocab_size - 1)
        self.index2word = tf.contrib.lookup.index_to_string_table_from_tensor(vocab_tensor, default_value='<UNK>')
        """
        self.word2index = {}
        self.index2word = {}
        
        v = ['<PAD>'] + vocab + ['<UNK>']
        for index, word in enumerate(v):
            self.word2index[word] = index
            self.index2word[index] = word
        """

        self.embedding_size = embedding_size

        self.embeddings = tf.get_variable(name + '_embeddings', initializer=tf.random_uniform([self.vocab_size, self.embedding_size], -0.1, 0.1), dtype=tf.float32)

    def get_embeddings(self):
        """
        returns the tensorflow Variable that contains the embedding weights
        """
        return self.embeddings

    def get_indexes_from_words_list(self, words):
        """
        words is a list of word, return a tensor of indexes
        """
        words = tf.constant(words)
        return self.word2index.lookup(words)

    def get_indexes_from_words_tensor(self, words):
        """
        words is a tensor of word, return a tensor of indexes
        """
        return self.word2index.lookup(words)

    def get_words_from_indexes(self, indexes):
        """
        from a tensor of integers, gives back the corresponding words
        """
        return self.index2word.lookup(indexes)

    def get_word_embeddings(self, words):
        """
        from a tensor of words to a tensor of embeddings
        """
        indexes = self.word2index.lookup(words)
        return tf.nn.embedding_lookup(self.embeddings, indexes)

    def get_word_embeddings_from_ids(self, word_ids):
        return tf.nn.embedding_lookup(self.embeddings, word_ids)

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class FixedEmbeddings(object):
    def __init__(self, tokenizer='space', language='en'):
        self.language = language
        language_model_name = get_language_model_name(language)

        import spacy
        self.nlp = spacy.load(language_model_name)
        if self.language == 'it' and False:
            # load the pretrained glove italian vectors, changing False on previous line
            # glove_wiki_it or glove_wiki_it_whitespace for old pretrained by http://hlt.isti.cnr.it/wordembeddings/
            self.nlp.vocab.vectors.from_disk('data/embeddings/glove_wiki_it/spacy_vectors_it')
            if self.nlp.vocab.vectors.shape[0] == 0:
                raise FileNotFoundError('impossible to find the embeddings file, run make create_it_embeddings')

        self.embedding_size = self.nlp.vocab.vectors.shape[1]
        if self.embedding_size == 0:
            # context vectors only
            self.embedding_size = 384

        print('tokenizer:', tokenizer, 'language_model_name:', language_model_name)
        if tokenizer == 'space':
            self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)


    def get_word_embeddings(self, words):

        def static_wrapper(words):
            return spacy_wrapper(self.embedding_size, self.language, self.nlp, words)

        result = tf.py_func(static_wrapper, [words], tf.float32, stateful=False, name='spacy_wrapper')
        shape = words.get_shape().as_list() + [self.embedding_size]
        result.set_shape(shape)
        return result


class FineTuneEmbeddings(FixedEmbeddings):
    def __init__(self, name, tokenizer='space', language='en'):
        super().__init__(tokenizer, language)

        self.fine_tune_embeddings = tf.get_variable(name + '_fine_tune', initializer=tf.constant(np.identity(self.embedding_size)), dtype=tf.float32)

    def get_word_embeddings(self, words):

        fixed_emb = super().get_word_embeddings(words)
        # linear transformation of embeddings
        # from a (time,batch,emb) 3d matrix reshape to (time*batch,emb) 2d
        fixed_emb = tf.reshape(fixed_emb, [-1, self.embedding_size])
        # multiply the matrix (time*batch, emb)x(emb,emb) = (time*batch,emb)
        result = tf.matmul(fixed_emb, self.fine_tune_embeddings)
        # and put again the result in 3d (time,batch,emb)
        result = tf.reshape(result, [words.shape.as_list()[0], -1, self.embedding_size])
        return result


def spacy_wrapper(embedding_size, language, nlp, words_numpy):
    embeddings_values = np.zeros([words_numpy.shape[0], words_numpy.shape[1], embedding_size], dtype=np.float32)
    for j, column in enumerate(words_numpy.T):
        # rebuild the sentence
        words = [w.decode('utf-8') for w in column]
        real_length = words.index('<EOS>')
        # special value for EOS
        embeddings_values[real_length,j,:] = np.ones((embedding_size))
        # remove padding words, embedding values have already been initialized to zero
        words = words[:real_length]
        if language == 'it':
            # TODO handle correctly uppercase/lowercase
            #words = [w.lower() for w in words]
            pass
        # put back together the sentence in order to get the word embeddings with context (only for languages without vectors)
        # TODO skip this if always word vectors, since if word vectors are part of the model, they are fixed and can get them simply by doing lookup
        # unless contextual vectors can be built also when vectors are there
        sentence = ' '.join(words)
        if language == 'en' or language == 'it':
            # only make_doc instead of calling nlp, much faster
            doc = nlp.make_doc(sentence)
        else:
            # other languages don't have pretrained word embeddings but use context vectors, really slower
            doc = nlp(sentence)
        # now get the vectors for each token
        for i, w in enumerate(doc):
            if i < real_length:
                if i >= words_numpy.shape[0]:
                    print('out of length', w)
                    print(sentence)
                else:
                    if not w.has_vector:
                        # TODO if oov:
                        #   try lowercase
                        #print('word', w, 'does not have a vector')
                        punctuations = '.?!,;:-_()[]{}\''
                        # TODO handle OOV punctuation marks without special case
                        if language == 'it' and w.text in punctuations:
                            punct_idx = punctuations.index(w.text)
                            embeddings_values[i,j,:] = np.ones((embedding_size))*punct_idx+2
                    else:
                        embeddings_values[i,j,:] = w.vector
                
    return embeddings_values


def get_language_model_name(language):
    if language == 'en':
        return 'en_vectors_web_lg'
    if language == 'it':
        return 'it_vectors_wiki_lg'

    return language
