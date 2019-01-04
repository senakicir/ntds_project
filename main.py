import numpy as np
import pygsp as pg
import pdb

from dataloader import *
from utils import *
from visualization import *
#from models import *
from error import error_func
from graph_analysis import Our_Graph

if __name__ == "__main__":
    adjacency, adjacency_pg = load_adjacency_matrix_from_npy()
    gt_labels = load_labels_from_npy()
    graph = Our_Graph(adjacency)
    #graph.compute_gradient()

    import pdb
    pdb.set_trace()
    plot_gt_labels(adjacency_pg, gt_labels)
