import numpy as np
import pdb
import pygsp as pg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Your code here.
def plot_gt_labels(graph, labels, name):
    graph.plot_signal(labels)
    plt.title('Groundtruth Labels')

    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    name += '_gt_clusters.png'
    plt.savefig("visualizations/" + name)
    plt.close()

def plot_confusion_matrix(predictions, gt, classes, name):
    our_confusion_matrix = confusion_matrix(gt, predictions)*100/len(gt)
    _, ax = plt.subplots()
    plt.title('Confusion Matrix')
    plt.imshow(our_confusion_matrix)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, our_confusion_matrix[i, j], ha="center", va="center", color="b")
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    name += '_confusion_matrix.png'
    plt.savefig("visualizations/" + name)
    plt.close()