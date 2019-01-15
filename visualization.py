import numpy as np
import pdb
import pygsp as pg
import matplotlib.pyplot as plt
import os

# Your code here.
def plot_gt_labels(graph, labels, name):
    graph.plot_signal(labels)
    plt.title('Groundtruth Labels')

    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    name += 'gt_clusters.png'
    plt.savefig("visualizations/" + name)
    plt.close()

def plot_confusion_matrix(our_confusion_matrix, classes, name):
    _, ax = plt.subplots()
    plt.title('Confusion Matrix')
    plt.imshow(our_confusion_matrix)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, '{:.2f}'.format(our_confusion_matrix[i, j]), ha="center", va="center", color="b")

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    name += 'confusion_matrix.png'
    plt.savefig("visualizations/" + name)
    plt.close()

def plot_errors_over_param(num_of_neigh_range, errors, name, model_name):
    _, ax = plt.subplots()
    plt.title('Error vs. param for ' + model_name)
    plt.plot(num_of_neigh_range, errors, marker = "^")

    plt.ylabel('error')
    if model_name == "KNN":
        plt.xlabel('number of neighbors')

    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    name += 'grid_search_' + model_name + '.png'
    plt.savefig("visualizations/" + name)
    plt.close()
