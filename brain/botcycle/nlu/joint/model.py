import tensorflow as tf
import numpy as np
from .data import spacy_wrapper

class RestoredModel(object):
    """
    Restores a model from a checkpoint
    """

    def __init__(self, model_path, embedding_size, language, nlp):

        # Step 1: restore the meta graph

        with tf.Graph().as_default() as graph:
            saver = tf.train.import_meta_graph(model_path + "model.ckpt.meta")
        
            self.graph = graph

            # get tensors for inputs and outputs by name
            self.decoder_prediction = graph.get_tensor_by_name('decoder_prediction:0')
            self.intent = graph.get_tensor_by_name('intent:0')
            self.intent_score = graph.get_tensor_by_name('intent_score:0')
            self.words_inputs = graph.get_tensor_by_name('words_inputs:0')
            self.encoder_inputs_actual_length = graph.get_tensor_by_name('encoder_inputs_actual_length:0')
            # redefine the py_func that is not serializable
            def static_wrapper(words):
                return spacy_wrapper(embedding_size, language, nlp, words)

            after_py_func = tf.py_func(static_wrapper, [self.words_inputs], tf.float32, stateful=False, name='spacy_wrapper')

            # Step 2: restore weights
            self.sess = tf.Session()
            self.sess.run(tf.tables_initializer())
            saver.restore(self.sess, model_path + "model.ckpt")


    def test(self, inputs):

        seq_in, length = list(zip(*[(sample['words'], sample['length']) for sample in inputs]))
        
        output_feeds = [self.decoder_prediction, self.intent, self.intent_score]
        feed_dict = {self.words_inputs: np.transpose(seq_in, [1, 0]), self.encoder_inputs_actual_length: length}

        results = self.sess.run(output_feeds, feed_dict=feed_dict)

        slots_batch, intent_batch, intent_score_batch = results
        for idx, slots in enumerate(slots_batch):
            slots_batch[idx] = np.array([slot.decode('utf-8') for slot in slots])
        for idx, intent in enumerate(intent_batch):
            intent_batch[idx] = intent.decode('utf-8')
        results = slots_batch, intent_batch, intent_score_batch
        return results