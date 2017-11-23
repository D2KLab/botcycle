import tensorflow as tf
import numpy as np

class EmbeddingsFromScratch(object):
  
    def __init__(self, vocab, embedding_size):
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

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -0.1, 0.1), dtype=tf.float32)

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
    def __init__(self, vocab=None, embedding_size=300, lang='en'):
        self.embedding_size = embedding_size
        if embedding_size != 300:
            raise ValueError('embedding size mismatch with precomputed embeddings!')

        import spacy
        self.nlp = spacy.load(lang)
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def get_word_embeddings(self, words):

        def spacy_wrapper(words_numpy):
            embeddings_values = np.zeros([words_numpy.shape[0], words_numpy.shape[1], self.embedding_size], dtype=np.float32)
            for j, column in enumerate(words_numpy.T):
                # build the sentence
                sentence = ' '.join([w.decode('utf-8') for w in column])
                sentence = sentence.replace('<','')
                sentence = sentence.replace('>','')
                # apostrophe problem -> custom tokenizer
                doc = self.nlp(sentence)
                #assert len(doc) is column.size
                # TODO problems when PAD or EOS are actual words!!
                for i, w in enumerate(doc):
                    embeddings_values[i,j,:] = w.vector
            return embeddings_values

        result = tf.py_func(spacy_wrapper, [words], tf.float32, stateful=False)
        shape = words.get_shape().as_list() + [self.embedding_size]
        result.set_shape(shape)
        return result
