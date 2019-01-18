from dataloader import *
from utils import *
from visualization import *
from models import *
from error import error_func
from graph_analysis import Our_Graph
from trainer import Trainer
from sklearn.metrics import confusion_matrix
import copy
import time as time

def cross_validation(model_ori, n_data, classes, K=5, name = ""):
    batch_size = n_data//K

    shuffled_ind = np.random.permutation(n_data)

    errors = np.zeros([K,])
    confusion_matrices = np.zeros([K, len(classes), len(classes)])
    for k in range(K):
        model = copy.deepcopy(model_ori)
        model.reset()
        idx_test = shuffled_ind[batch_size*k:batch_size*(k+1)]
        idx_tr = np.concatenate([shuffled_ind[0:batch_size*k], shuffled_ind[batch_size*(k+1):-1]], axis=0)

        model.train(idx_tr)
        model.classify(idx_test)
        confusion_matrices[k, :, :], errors[k] = model.accuracy(classes)
        #print('Iter {0:d} Percentage Error: {1:.2f}'.format(k, errors[k]))

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    overall_confusion_matrix = np.mean(confusion_matrices, axis=0)
    plot_confusion_matrix(overall_confusion_matrix, classes, name)
    return mean_error, std_error


def evaluate_transductive(model_ori, idx_train, idx_test, classes, name = ""):
    model = copy.deepcopy(model_ori)

    model.train(idx_train)
    model.classify(idx_test)
    confusion_matrix, error = model.accuracy(classes)

    plot_confusion_matrix(confusion_matrix, classes, name)
    return error


def simple_test(model_ori, n_data, classes, name = ""):
    model = copy.deepcopy(model_ori)
    model.load_pretrained()

    idx_test = np.array(list(range(n_data)))
    model.classify(idx_test)

    confusion_matrix, error = model.accuracy(classes)

    plot_confusion_matrix(confusion_matrix, classes, name)
    return error
