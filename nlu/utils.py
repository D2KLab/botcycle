import numpy as np
import matplotlib.pyplot as plt

def plot_f1_history(file_name, train_f1_history, test_f1_history=np.array([])):
    plt.clf()
    plt.plot(train_f1_history)
    if test_f1_history.shape[0] > 0:
        plt.plot(test_f1_history)
        plt.legend(['train', 'test'], loc='lower right')
    else:
        plt.legend(['train'], loc='lower right')

    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epochs')
    print(file_name)
    plt.savefig(file_name)