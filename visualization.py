import numpy as np
import pdb
import pygsp as pg
import matplotlib.pyplot as plt

# Your code here.
def plot_gt_labels(graph, labels):
    graph.plot_signal(labels)
    plt.title('Groundtruth Labels')
    plt.savefig('gt_clusters.png')
    plt.close()