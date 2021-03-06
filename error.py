import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb

from utils import *

def error_func(gt_labels, pred_labels):

    assert gt_labels.shape == pred_labels.shape, "Groundtruth labels should have the same shape as the prediction labels"

    if len(gt_labels.shape) == 2:
        gt_labels = gt_labels.squeeze(1)
    if len(pred_labels.shape) == 2:
        pred_labels = pred_labels.squeeze(1)

    errors = sum(gt_labels != pred_labels)
    error_rate = (errors/gt_labels.shape[0]) * 100
    return error_rate

def accuracy_prob(pred, gt):
    #When the prediction has probabilities for each class and
    #the gt keep the location of the true class use this function.
    pred = pred.max(1)[1].type_as(gt)
    acc = pred.eq(gt).double()
    acc_rate = (acc.sum() / gt.shape[0]) * 100
    return acc_rate