from . import __plot_helpers as pl
from .__data_helpers import *
from .__spacorr import compute_spatial_autocorr, Morans_I, spatial_regression_test
from .__pred_pad import complete_unseen_expression, annotation_propagate
from .__cluster_helpers import mclust_R
from .__homophily import node_homophily, edge_homophily


def init_mclust_R():
    import rpy2.robjects as robjects
    robjects.r.library("mclust")