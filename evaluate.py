from dataloader import *
from utils import *
from visualization import *
from models import *
from error import error_func
from graph_analysis import Our_Graph
from trainer import Trainer

def cross_validation(features, labels, model, K=5):
    n_data = features.shape[0]
    batch_size = n_data//K

    shuffled_ind = np.random.permutation(n_data)
    shuffled_feat = features[shuffled_ind, :]
    shuffled_labels = labels[shuffled_ind, :]

    errors = np.zeros([K,])
    for k in range(K):
        features_test = shuffled_feat[batch_size*k:batch_size*(k+1),:]
        features_tr = np.concatenate([shuffled_feat[0:batch_size*k,:], shuffled_feat[batch_size*(k+1):-1,:]], axis=0)

        gt_labels_test = shuffled_labels[batch_size*k:batch_size*(k+1),0]
        gt_labels_tr = np.concatenate([shuffled_labels[0:batch_size*k,0], shuffled_labels[batch_size*(k+1):-1,0]], axis=0)

        model.train(features_tr, gt_labels_tr)
        predicted_classes_svm = model.classify(features_test)
        #plot_confusion_matrix(predicted_classes_svm, gt_labels_test, ['Hip-Hop', 'Rock'], "")
        errors[k] = error_func(gt_labels_test, predicted_classes_svm)
        print('Iter {0:d} Percentage Error: {1:.2f}'.format(k, errors[k]))
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error