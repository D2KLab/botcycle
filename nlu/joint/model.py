# coding=utf-8
# @author: cer
import sys
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from .embeddings import EmbeddingsFromScratch, FixedEmbeddings

flatten = lambda l: [item for sublist in l for item in sublist]

class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, vocabs, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.words_inputs = tf.placeholder(tf.string, [input_steps, batch_size], name="word_inputs")
        #self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size], name='encoder_inputs')
        # Enter the actual length of each sentence, in addition to padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.string, [batch_size, input_steps],
                                              name='decoder_targets')
        self.intent_targets = tf.placeholder(tf.string, [batch_size],
                                             name='intent_targets')

        self.vocabs = vocabs

    def build(self):

        input_vocab, slot_vocab, intent_vocab = self.vocabs
        # then create the embeddings and mapper for each one of them

        self.input_embedding_size = 300
        # choose if input words are trained as part of the model from scratch, or come precomputed
        self.wordsEmbedder = FixedEmbeddings(self.input_embedding_size)
        #self.wordsEmbedder = EmbeddingsFromScratch(input_vocab, self.input_embedding_size)
        self.slotEmbedder = EmbeddingsFromScratch(slot_vocab, self.embedding_size)
        self.intentEmbedder = EmbeddingsFromScratch(intent_vocab, self.embedding_size)

        # the embedded inputs
        self.encoder_inputs_embedded = self.wordsEmbedder.get_word_embeddings(self.words_inputs)

        # Encoder

        # Use a single LSTM cell
        encoder_f_cell = LSTMCell(self.hidden_size)
        encoder_b_cell = LSTMCell(self.hidden_size)
        # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
        # The size of the following four variables：T*B*D，T*B*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        # Decoder
        decoder_lengths = self.encoder_inputs_actual_length
        self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
        intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")

        # Seek intent
        intent_logits = tf.add(tf.matmul(encoder_final_state_h, intent_W), intent_b)
        # intent_prob = tf.nn.softmax(intent_logits)
        intent_id = tf.argmax(intent_logits, axis=1)
        self.intent = self.intentEmbedder.get_words_from_indexes(intent_id)

        # generate a tensor of batch_size * 'O' for start of sentence
        sos_time_slice = tf.constant(np.tile('O', self.batch_size))
        sos_step_embedded = self.slotEmbedder.get_word_embeddings(sos_time_slice)
        # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        pad_time_slice = tf.constant(np.tile('<PAD>', self.batch_size))
        pad_step_embedded = self.slotEmbedder.get_word_embeddings(pad_time_slice)
        print("pad_step_embedded", pad_step_embedded)

        def initial_fn():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # Logit argmax as a sample
            print("outputs", outputs)
            # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
            # print("slot output_logits: ", output_logits)
            # prediction_id = tf.argmax(output_logits, axis=1)
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            # The output category on the last time node, get embedding and then enter as the next time node
            pred_embedding = self.slotEmbedder.get_word_embeddings_from_ids(sample_ids)
            # The input is h_i+o_{i-1}+c_i
            next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            #next_inputs.set_shape(next_input.shape)
            next_inputs = next_input
            next_state = state
            return elements_finished, next_inputs, next_state

        my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size, memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)
                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_size)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.slot_size, reuse=reuse
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size))
                # initial_state=encoder_final_state)
                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=True,
                    impute_finished=True, maximum_iterations=self.input_steps
                )
                return final_outputs

        outputs = decode(my_helper, 'decode')
        print("outputs: ", outputs)
        print("outputs.rnn_output: ", outputs.rnn_output)
        print("outputs.sample_id: ", outputs.sample_id)
        # weights = tf.to_float(tf.not_equal(outputs[:, :-1], 0))
        self.decoder_prediction = self.slotEmbedder.get_words_from_indexes(tf.to_int64(outputs.sample_id))
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
        decoder_targets_ids = self.slotEmbedder.get_indexes_from_words_tensor(self.decoder_targets)
        self.decoder_targets_time_majored = tf.transpose(decoder_targets_ids, [1, 0])
        self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
        print("decoder_targets_true_length: ", self.decoder_targets_true_length)
        # Define mask so padding does not count towards loss calculation
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
        # Define slot marked loss
        loss_slot = tf.contrib.seq2seq.sequence_loss(
            outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)
        # Define intent category loss
        intent_ids_targets = self.intentEmbedder.get_indexes_from_words_tensor(self.intent_targets)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(intent_ids_targets, depth=self.intent_size, dtype=tf.float32),
            logits=intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)

        self.loss = loss_slot + loss_intent
        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.grads, self.vars))
        # self.train_op = optimizer.minimize(self.loss)
        # train_op = layers.optimize_loss(
        #     loss, tf.train.get_global_step(),
        #     optimizer=optimizer,
        #     learning_rate=0.001,
        #     summaries=['loss', 'learning_rate'])

    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        # print(np.transpose(unziped[0], [1, 0]))
        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                            self.intent, self.mask, self.slot_W]
            feed_dict = {self.words_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1],
                         self.decoder_targets: unziped[2],
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.decoder_prediction, self.intent]
            feed_dict = {self.words_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)
        if mode in ['test']:
            slots_batch, intent_batch = results
            for idx, slots in enumerate(slots_batch):
                slots_batch[idx] = np.array([slot.decode('utf-8') for slot in slots])
            for idx, intent in enumerate(intent_batch):
                intent_batch[idx] = intent.decode('utf-8')
            results = slots_batch, intent_batch
        return results
