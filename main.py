import numpy as np
import pygsp as pg
import pdb

from dataloader import *
from utils import *
from visualization import *
from models import *

if __name__ == "__main__":
    adjacency, adjacency_pg = load_adjacency_matrix_from_npy()
    gt_labels = load_labels_from_npy()

    plot_gt_labels(adjacency_pg, gt_labels)
