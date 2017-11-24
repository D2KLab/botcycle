import sys
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from .embeddings import EmbeddingsFromScratch, FixedEmbeddings

flatten = lambda l: [item for sublist in l for item in sublist]

class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocabs, batch_size=16):
        # save the parameters
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # also save the vocabularies, used by embedders
        self.vocabs = vocabs
        self.input_embedding_size = 300

        # define the placeholders for inputs to the graph
        # the input words are a tensor of type string.
        # In this way the one_hot encoding stuff and embeddings are managed by the embedding classes.
        # This makes the input always to be strings, both when the embeddings are part of the model
        # both when are precomputed
        self.words_inputs = tf.placeholder(tf.string, [input_steps, batch_size], name="word_inputs")
        # This placeholder is for the actual length of each sentence, used in decoding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size], name='encoder_inputs_actual_length')
        # Placeholder for the output sequence, used in train mode as truth value
        self.decoder_targets = tf.placeholder(tf.string, [batch_size, input_steps], name='decoder_targets')
        # Placeholder for the output intent, used in train mode as truth value
        self.intent_targets = tf.placeholder(tf.string, [batch_size], name='intent_targets')
        

    def build(self):
        # unpack the vocabularies
        input_vocab, slot_vocab, intent_vocab = self.vocabs

        # then create the embeddings and mapper (one-hot index to words and viceversa) for each one of them
        # For input words embedder, can choose between EmbeddingsFromScratch, FixedEmbeddings:
        # choose if input words are trained as part of the model from scratch, or come precomputed
        self.wordsEmbedder = FixedEmbeddings(self.input_embedding_size)
        #self.wordsEmbedder = EmbeddingsFromScratch(input_vocab, self.input_embedding_size)
        self.slotEmbedder = EmbeddingsFromScratch(slot_vocab, self.embedding_size)
        self.intentEmbedder = EmbeddingsFromScratch(intent_vocab, self.embedding_size)

        # the embedded inputs
        self.encoder_inputs_embedded = self.wordsEmbedder.get_word_embeddings(self.words_inputs)

        # Encoder

        # Definition of cells used for bidirectional RNN encoder
        encoder_f_cell = LSTMCell(self.hidden_size)
        encoder_b_cell = LSTMCell(self.hidden_size)
        # Bidirectional RNN
        # The size of the following four variables：T*B*D，T*B*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)

        # Encoder outputs

        # The encoder outputs are the concatenation of the outputs of each direction.
        # The concatenation is done on the third dimension. Dimensions: (time, batch, hidden_size)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        # Also concatenate things for the final state. Dimensions: (batch, hidden_size)
        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
        self.encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)


        # Intent output
        
        # Define the weights and biases to perform the output projection on the intent output
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size, self.intentEmbedder.vocab_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
        intent_b = tf.Variable(tf.zeros([self.intentEmbedder.vocab_size]), dtype=tf.float32, name="intent_b")

        # Intent RNN
        decoder_intent_cell = LSTMCell(self.hidden_size)

        decoder_intent_outputs, decoder_intent_final_state = tf.nn.dynamic_rnn(decoder_intent_cell,
                                            inputs=encoder_outputs,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)

        # perform the feed-forward layer
        intent_logits = tf.add(tf.matmul(decoder_intent_final_state.h, intent_W), intent_b)
        # take the argmax
        intent_id = tf.argmax(intent_logits, axis=1)
        # and translate to the corresponding string
        self.intent = self.intentEmbedder.get_words_from_indexes(intent_id)


        # Slot label decoder

        decoder_lengths = self.encoder_inputs_actual_length

        # Initial values to provide to the decoding stage
        # generate a tensor of batch_size * 'O' for start of sentence.
        # This value will be passed to first iteration of decoding in place of the previous slot label
        sos_time_slice = tf.constant(np.tile('O', self.batch_size))

        # the following functions are used by the CustomHelper
        def initial_fn():
            """
            defines how to provide the input to the decoder RNN cell at time 0
            """
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            # get the embedded representation of the initial fake previous-output-label
            sos_step_embedded = self.slotEmbedder.get_word_embeddings(sos_time_slice)
            # then concatenate it with the encoder output at time 0
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            """
            defines how to sample from the output of the RNN cell
            """
            # take the argmax from the logits
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            """
            defines how to provide the input to the RNN cell at timesteps>0
            """
            # From the last output, represented by sample_ids, get its embedded value
            pred_embedding = self.slotEmbedder.get_word_embeddings_from_ids(sample_ids)
            # Now concatenate it with the output of the decoder at the current timestep.
            # This is the new input to the RNN cell
            next_inputs = tf.concat((pred_embedding, encoder_outputs[time]), 1)
            # Establish which samples in the batch have already finished the decoding
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            # don't modify the state
            next_state = state
            return elements_finished, next_inputs, next_state

        # Build the helper with the declared functions
        my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        # Decoding function
        def decode(helper, scope, reuse=None):
            # define an isolated scope
            with tf.variable_scope(scope, reuse=reuse):
                # Get the memory representation (for the attention) by making the
                # encoder outputs dimensions from (time, batch, hidden_size) to (batch, time, hidden_size)
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
                # Use the BahdanauAttention on the memory
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size, memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)
                # The decoding LSTM cell
                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                # that gets wrapped inside the attention mechanism
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_size)
                # and gets wrapped inside a output projection wrapper (weights+biases),
                # to have an output with logits on the slot labels dimension
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.slotEmbedder.vocab_size, reuse=reuse
                )
                # Define the decoder by combining the helper with the RNN cell
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size))
                # And finally perform the decode
                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=True,
                    impute_finished=True, maximum_iterations=self.input_steps
                )
                return final_outputs

        outputs = decode(my_helper, 'decode')
        
        # Now from the slot decoder outputs, get the corresponding output word (slot label, from ids to words)
        self.decoder_prediction = self.slotEmbedder.get_words_from_indexes(tf.to_int64(outputs.sample_id))
        # Get some informations on the performed decoding: the maximum number of steps done in the batch
        decoder_max_steps, _, _ = tf.unstack(tf.shape(outputs.rnn_output))


        # Losses requirements: get comparable tensors from graph and from target values

        # For slot filling
        # Now on the decoder targets (used in training only), get their ids (from words to ids)
        decoder_targets_ids = self.slotEmbedder.get_indexes_from_words_tensor(self.decoder_targets)
        # Swap the dimensions: from (batch, time) to (time, batch)
        self.decoder_targets_time_majored = tf.transpose(decoder_targets_ids, [1, 0])
        # Truncate them on the actual decoding maximum number of steps (to have same length as decoder outputs)
        self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
        # Define mask so padding does not count towards loss calculation
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))

        # For the intent
        intent_ids_targets = self.intentEmbedder.get_indexes_from_words_tensor(self.intent_targets)


        # Losses definitions
        # for the slots, using builtin sequence_loss
        loss_slot = tf.contrib.seq2seq.sequence_loss(
            outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)
        # For the intent, using cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(intent_ids_targets, depth=self.intentEmbedder.vocab_size, dtype=tf.float32),
            logits=intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)
        # Combine the losses
        self.loss = loss_slot + loss_intent
        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        #print("vars for loss function: ", self.vars)
        # Clip gradients to prevent exploding ones
        gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.grads, self.vars))


    def step(self, sess, mode, train_batch):
        """do a step on the current batch"""
        if mode not in ['train', 'test']:
            print('mode is not supported', file=sys.stderr)
            sys.exit(1)
        unziped = list(zip(*train_batch))
        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                            self.intent, self.mask]
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
