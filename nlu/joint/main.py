import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from . import data
from .model import Model
from . import metrics

# maximum length of sentences
input_steps = 50
# embedding size for labels
embedding_size = 64
# size of LSTM cells
hidden_size = 100
# size of batch
batch_size = 16
# number of training epochs
epoch_num = 50


def get_model(vocabs, tokenizer):
    model = Model(input_steps, embedding_size, hidden_size, vocabs, batch_size)
    model.build(tokenizer)
    return model


def train(is_debug=False):
    # load the train and dev datasets
    test_data, train_data = data.load_data('nlu-benchmark')
    # fix the random seeds
    random_seed_init(len(test_data['data']))
    # preprocess them to list of training/test samples
    # a sample is made up of a tuple that contains
    # - an input sentence (list of words --> strings, padded)
    # - the real length of the sentence (int) to be able to recognize padding
    # - an output sequence (list of IOB annotations --> strings, padded)
    # - an output intent (string)
    training_samples = data.data_pipeline(train_data)
    test_samples = data.data_pipeline(test_data)
    # get the vocabularies for input, slot and intent
    vocabs = data.get_vocabularies(training_samples)
    # and get the model
    model = get_model(vocabs, training_samples['meta']['tokenizer'])
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    # initialize the required parameters
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # initialize the history that will collect some measures
    history = {
        'intent': np.zeros((epoch_num)),
        'slot': np.zeros((epoch_num))
    }
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(data.getBatch(batch_size, training_samples['data'])):
            # perform a batch of training
            _, loss, decoder_prediction, intent, mask = model.step(sess, "train", batch)
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10.0
                #print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                print('.', end=' ')
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # test each epoch once
        pred_slots = []
        pred_intents = []
        true_intents = []
        for j, batch in enumerate(data.getBatch(batch_size, test_samples['data'])):
            decoder_prediction, intent = model.step(sess, "test", batch)
            # from time-major matrix to sample-major
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                print("Input Sentence        : ", batch[index]['words'])
                print("Slot Truth            : ", batch[index]['slots'])
                print("Slot Prediction       : ", decoder_prediction[index])
                print("Intent Truth          : ", batch[index]['intent'])
                print("Intent Prediction     : ", intent[index])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            #print("pred_intents", pred_intents, "intent", intent)
            pred_intents.extend(intent)
            true_intent = [sample['intent'] for sample in batch]
            true_intents.extend(true_intent)
            #print("true_intents", true_intents)
            # print("slot_pred_length: ", slot_pred_length)
            true_slot = np.array([sample['slots'] for sample in batch])
            true_length = np.array([sample['length'] for sample in batch])
            true_slot = true_slot[:, :slot_pred_length]
            # print(np.shape(true_slot), np.shape(decoder_prediction))
            # print(true_slot, decoder_prediction)
            slot_acc = metrics.accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = metrics.accuracy_score(true_intent, intent)
            #print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
        pred_slots_a = np.vstack(pred_slots)
        # print("pred_slots_a: ", pred_slots_a.shape)
        true_slots_a = np.array([sample['slots'] for sample in test_samples['data']])[:pred_slots_a.shape[0]]
        f1_intents = metrics.f1_for_intents(pred_intents, true_intents)
        f1_slots = metrics.f1_for_sequence_batch(true_slots_a, pred_slots_a)
        # print("true_slots_a: ", true_slots_a.shape)
        print("F1 score SEQUENCE for epoch {}: {}".format(epoch, f1_slots))
        print("F1 score INTENTS for epoch {}: {}".format(epoch, f1_intents))
        history['intent'][epoch] = f1_intents
        history['slot'][epoch] = f1_slots

    metrics.plot_f1_history('f1.png', history)
    print(history)

def random_seed_init(seed):
    random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == '__main__':
    train()