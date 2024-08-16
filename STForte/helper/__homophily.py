import numpy as np
import pandas as pd
from typing import Union

def node_homophily(labels: Union[np.ndarray,pd.Categorical], adjacency: np.ndarray):
    # one-hot encoding of the node labels
    categories = list(set(labels))
    labels = np.array([categories.index(i) for i in labels])
    one_hot_labels = np.zeros((labels.shape[0],len(categories)))
    for i in range(len(labels)):
        one_hot_labels[i,labels[i]] = 1

    # calculate the label sum
    label_sum = adjacency.dot(one_hot_labels)
    label_sum_same = (label_sum * one_hot_labels).sum(axis=1)
    label_sum_all = label_sum.sum(axis=1)
    h_node = (label_sum_same[label_sum_all!=0] / label_sum_all[label_sum_all!=0]).sum() / labels.shape[0]
    return h_node

def edge_homophily(labels: Union[np.ndarray,pd.Categorical], adjacency: np.ndarray):
    labels = np.array(labels)
    categories = list(set(labels))
    labels = np.array([categories.index(i) for i in labels]).reshape(-1,1)
    edge_list = np.argwhere(adjacency==1)
    label_1, label_2 = labels[edge_list[:,0]], labels[edge_list[:,1]]
    label_sum_same = ((label_1 - label_2) == 0).sum()
    return label_sum_same / edge_list.shape[0]