# coding=utf-8
# @author: cer
import tensorflow as tf
from .data import *
# from model import Model
from .model import Model
from .my_metrics import *
from tensorflow.python import debug as tf_debug

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 871
slot_size = 122
intent_size = 22
epoch_num = 20


def get_model(train_data_ed):
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, train_data_ed, batch_size, n_layers)
    model.build()
    return model


def train(is_debug=False):
    train_data = open("data/atis/source/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("data/atis/source/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    #word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    #    get_info_from_training_data(train_data_ed)

    #index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    #index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

    model = get_model(train_data_ed)
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # print(tf.trainable_variables())

    history = {
        'intent': np.zeros((epoch_num)),
        'slot': np.zeros((epoch_num))
    }
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, train_data_ed)):
            # perform a batch of training
            _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)
            # if i == 0:
            #     index = 0
            #     print("training debug:")
            #     print("input:", list(zip(*batch))[0][index])
            #     print("length:", list(zip(*batch))[1][index])
            #     print("mask:", mask[index])
            #     print("target:", list(zip(*batch))[2][index])
            #     # print("decoder_targets_one_hot:")
            #     # for one in decoder_targets_one_hot[index]:
            #     #     print(" ".join(map(str, one)))
            #     print("decoder_logits: ")
            #     for one in decoder_logits[index]:
            #         print(" ".join(map(str, one)))
            #     print("slot_W:", slot_W)
            #     print("decoder_prediction:", decoder_prediction[index])
            #     print("intent:", list(zip(*batch))[3][index])
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10.0
                print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # test each epoch once
        pred_slots = []
        pred_intents = []
        true_intents = []
        for j, batch in enumerate(getBatch(batch_size, test_data_ed)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                print("Input Sentence        : ", batch[index][0])
                print("Slot Truth            : ", batch[index][2])
                print("Slot Prediction       : ", decoder_prediction[index])
                print("Intent Truth          : ", batch[index][3])
                print("Intent Prediction     : ", intent[index])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            #print("pred_intents", pred_intents, "intent", intent)
            pred_intents.extend(intent)
            true_intents.extend(list(zip(*batch))[3])
            #print("true_intents", true_intents)
            # print("slot_pred_length: ", slot_pred_length)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            # print(np.shape(true_slot), np.shape(decoder_prediction))
            # print(true_slot, decoder_prediction)
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
            print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
        pred_slots_a = np.vstack(pred_slots)
        # print("pred_slots_a: ", pred_slots_a.shape)
        true_slots_a = np.array(list(zip(*test_data_ed))[2])[:pred_slots_a.shape[0]]
        f1_intents = f1_for_intents(pred_intents, true_intents)
        f1_slots = f1_for_sequence_batch(true_slots_a, pred_slots_a)
        # print("true_slots_a: ", true_slots_a.shape)
        print("F1 score SEQUENCE for epoch {}: {}".format(epoch, f1_slots))
        print("F1 score INTENTS for epoch {}: {}".format(epoch, f1_intents))
        history['intent'][epoch] = f1_intents
        history['slot'][epoch] = f1_slots

    plot_f1_history('f1.png', history)


def test_data():
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    batch = next(getBatch(batch_size, index_test))
    unziped = list(zip(*batch))
    print("word num: ", len(word2index.keys()), "slot num: ", len(slot2index.keys()), "intent num: ",
          len(intent2index.keys()))
    print(np.shape(unziped[0]), np.shape(unziped[1]), np.shape(unziped[2]), np.shape(unziped[3]))
    print(np.transpose(unziped[0], [1, 0]))
    print(unziped[1])
    print(np.shape(list(zip(*index_test))[2]))


if __name__ == '__main__':
    # train(is_debug=True)
    # test_data()
    train()