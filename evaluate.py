from dataloader import *
from utils import *
from visualization import *
from models import *
from error import error_func
from graph_analysis import Our_Graph
from trainer import Trainer
from sklearn.metrics import confusion_matrix
import copy

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
        #confusion_matrices[k, :, :] = confusion_matrix(gt_labels_test, predicted_classes)
        errors[k] = model.accuracy()
        #print('Iter {0:d} Percentage Error: {1:.2f}'.format(k, errors[k]))

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    overall_confusion_matrix = np.mean(confusion_matrices, axis=0)
    plot_confusion_matrix(overall_confusion_matrix, classes, name)
    return mean_error, std_error


def grid_search_for_param(features, labels, model, model_name, classes, K=5, name = ""):
    if model_name == "KNN":
        num_of_neigh_range = list(range(10,100,2))
        errors = np.zeros([len(num_of_neigh_range), ])
        for ind, num_of_neigh in enumerate(num_of_neigh_range):
            model.reset(num_of_neigh)
            mean_error, _ = cross_validation(features, labels, model, K, classes, name)
            errors[ind] = mean_error
        min_index = np.argmin(errors)
        print('Minimum error of KNN is {:.2f} with {:d} number of neighbours.'.format(errors_arr[min_index], num_of_neigh_range[min_index]))
        plot_errors_over_param(num_of_neigh_range, errors, name, model_name)

    if model_name == "Random_Forest":
        n_estimators_range = list(range(50, 200, 25))
        max_depth_range = list(range(1, 4, 1))
        errors = np.zeros([len(n_estimators_range), len(max_depth_range)])
        for i, n_estimators in enumerate(n_estimators_range):
            for j, max_depth in enumerate(max_depth_range):
                model.reset(n_estimators, max_depth)
                mean_error, _ = cross_validation(features, labels, model, K, classes, name)
                errors[i, j] = mean_error

        errors_arr = np.array(errors)
        min_index = np.unravel_index(np.argmin(errors_arr), errors_arr.shape)
        print('Minimum error of RF is {:.2f} with {:d} number of estimators and {:d} depth.'.format(errors_arr[min_index], n_estimators_range[min_index[0]], max_depth_range[1]))
