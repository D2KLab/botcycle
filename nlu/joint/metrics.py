import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def accuracy_score(true_data, pred_data, true_length=None):
    true_data = np.array(true_data)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape
    if true_length is not None:
        val_num = np.sum(true_length)
        assert val_num != 0
        res = 0
        for i in range(true_data.shape[0]):
            res += np.sum(true_data[i, :true_length[i]] == pred_data[i, :true_length[i]])
    else:
        val_num = np.prod(true_data.shape)
        assert val_num != 0
        res = np.sum(true_data == pred_data)
    res /= float(val_num)
    return res


def get_data_from_sequence_batch(true_batch, pred_batch, eos_token):
    """Extract data from a batch of sequences：
    [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]"""
    true_ma = []
    pred_ma = []
    for idx, true in enumerate(true_batch):
        where = true.tolist()
        lentgth = where.index(eos_token)
        true = true[:lentgth]
        pred = pred_batch[idx][:lentgth]
        true_ma.extend(true.tolist())
        pred_ma.extend(pred.tolist())
    return true_ma, pred_ma


def f1_for_sequence_batch(true_batch, pred_batch, average="micro", eos_token='<EOS>'):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, eos_token)
    labels = list(set(true))
    return f1_score(true, pred, labels=labels, average=average)


def accuracy_for_sequence_batch(true_batch, pred_batch, eos_token='<EOS>'):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, eos_token)
    return accuracy_score(true, pred)

def f1_for_intents(true, pred, average="micro"):
    return f1_score(true, pred, average=average)

def plot_f1_history(file_name, history):
    plt.clf()
    for scores in history.values():
        plt.plot(scores)
    
    plt.legend(list(history.keys()), loc='lower right')
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epochs')
    plt.grid()
    print(file_name)
    plt.savefig(file_name)
