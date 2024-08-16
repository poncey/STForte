import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union

import scipy
import scipy.stats as stats
from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix

from sklearn.neighbors import kneighbors_graph
from scanpy.metrics._morans_i import _morans_i
from scanpy.metrics._gearys_c import _gearys_c
from anndata._core.anndata import AnnData

import rpy2
import rpy2.robjects as r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import Vector, Matrix

from ..utils.__data_utils import create_graph


def compute_spatial_autocorr(adata: AnnData, connectivity: Union[str, csr_matrix, np.ndarray], 
                             use_layer: str = None,
                             mode: str = "moran",
                             gene_set: Union[list, pd.Index] = None, 
                             cutoff: float = 0.5, 
                             two_tailed: bool = False,
                             copy: bool = False):
    """Compute spatial autocorrelation for the gene sets of ST data.

    Args:
        adata (AnnData): Anndata input as the spatial transcriptomics data.
        connectivity (Union[str, csr_matrix, np.ndarray]): Connectivity matrix. It can be string for the indexer of adata.obsp or csr_matrix/ndarray 
        for the matrix.
        use_layer (str, optional): Which layer to use. If None, Use adata.X. Defaults to None.
        mode (str, optional): Mode of autocorr. It can be "moran" for Moran's I or "geary" for Geary's C. Defaults to "moran".
        gene_set (Union[list, pd.Index], optional): List of genes to compute autocorr. If None, computation performed on all genes. Defaults to None.
        cutoff (float, optional): Cut-off of the connectivity matrix. If None, no cut-off will be performed. Defaults to 0.5.
        two_tailed (bool, optional): whether to use two-tailed test for generating p-value. Defaults to False.
        copy (bool, optional): If True, a new copy DataFrame of autocorr will be returned. Otherwise It will be joined in adata.uns. Defaults to False.
    
    Raises:
        ValueError: Wrong identifier of the 'mode'.

    Returns:
        df_autocorr (DataFramme): If copy is not True, the autocorr result will be returned as DataFrame format.
    """
    # Get gene values to investigate 
    if use_layer is not None:
        vals = adata.layers[use_layer]
    else:
        vals = adata.X
    if gene_set is not None:
        indexer = adata.var_names.get_indexer(gene_set)
        vals = vals[:, indexer]
    else:
        gene_set = adata.var_names
    vals = vals.T
    
    # Get connectivity matrix
    if isinstance(connectivity, str):
        A = adata.obsp[connectivity].copy()
    else:
        A = connectivity.copy()
    np.fill_diagonal(A, 0)
    # set cutoff
    if cutoff is not None:
        A[A <= 0.5] = 0
    A = csr_matrix(A)
    
    # calculate autocorr index   
    if mode == "moran":
        autocorr = _morans_i(A, vals)
    elif mode == "geary":
        autocorr = _gearys_c(A, vals)
    else:
        raise ValueError("Wrong mode: \"{:s}\". It shoulbe only be \"moran\" for Morans'I or \"geary\" for Geary's C.".format(mode))
    
    # calculate pval
    expect_val = -1.0 / (adata.shape[0] - 1) if mode == "moran" else 1.0
    pvals, _ = _analytic_pval(autocorr, A, expect_val, two_tailed=two_tailed)
    
    df_autocorr = pd.DataFrame({mode: autocorr, "pval":pvals}, index=gene_set).sort_values(by=mode, ascending=False)
    if copy is True:
        return df_autocorr
    else:
        adata.uns["STForte_AUTOCORR_{:s}".format(mode)] = df_autocorr
    

def _analytic_pval(score, g, expect_val, two_tailed: bool = False):

    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    n2 = n * n
    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02

    Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - expect_val) / seScore_norm
    p_norm = np.empty(score.shape)
    p_norm[z_norm > 0] = 1 - stats.norm.cdf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = stats.norm.cdf(z_norm[z_norm <= 0])

    if two_tailed:
        p_norm *= 2.0

    return p_norm, Vscore_norm

def _g_moments(w):
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t)
    s1 = t2.sum() / 2.0

    # s2
    s2array = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2

def Morans_I(vals: np.ndarray, spatial: np.ndarray, k: int=None):
    # calculate spatial weights
    if k is not None:
        weights = kneighbors_graph(spatial,n_neighbors=k,mode='distance').A
    else:
        weights = distance_matrix(spatial,spatial)
    weights[weights!=0] = 1/weights[weights!=0]
    
    # calcualte deviation
    deviation = vals - vals.mean(axis=0)

    # calculate Moran's I index
    n, n_vals = vals.shape
    moran_i, z_score, p_value = list(), list(), list()
    for i in range(n_vals):
        v = deviation[:,i]
        s0 = weights.sum()
        s1 = ((weights+weights.T)**2 / 2).sum()
        s2 = ((weights.sum(axis=0) + weights.sum(axis=1))**2).sum()
        D = (v**4).sum() / ((v**2).sum())**2
        A = n*((n**2 - 3*n + 3)*s1 - n*s2 + 3*s0**2)
        B = D*((n**2 - n)*s1 - 2*n*s2 + 6*s0**2)
        C = (n-1)*(n-2)*(n-3)*s0**2
        I = n*(v.T.dot(weights).dot(v)/v.T.dot(v))/s0
        moran_i.append(I)

        e_i = -1 / (n-1)
        e_i_2 = (A-B) / C
        v_i = e_i_2 - e_i**2
        z_i = (I-e_i)/np.sqrt(v_i)
        z_score.append(z_i)
        
        # print(s0,s1,s2,A,B,C,D,I)
        
        if z_i > 0:
            p = 1 - norm.cdf(z_i)
        else:
            p = norm.cdf(z_i)
        p_value.append(p)

    return np.array(moran_i), np.array(z_score), np.array(p_value) 


def r_lm(expression, cal_f=False):
    stats = importr('stats')
    base = importr('base')
    dataframe = pd.DataFrame(expression, columns=['x','y'])
    pandas2ri.activate()
    r.globalenv['dataframe'] = dataframe
    M = stats.lm('y~x', data=base.as_symbol('dataframe'))
    coe = pd.DataFrame(base.summary(M).rx2('coefficients'),
                  columns=['Estimate','Std.Error','t value','Pr(>|t|)'],
                  index=['(Intercept)','x']) 
    if cal_f:
        f_stat = pd.DataFrame(base.summary(M).rx2('fstatistic').reshape(1,3),
                      columns=['value','numdf','dendf'])
        f_p_value = 1 - scipy.stats.f.cdf(f_stat.to_numpy()[:,0],dfn=f_stat.to_numpy()[:,1],dfd=f_stat.to_numpy()[:,2])
        return coe, f_stat, f_p_value[0]
    return coe


def spatial_regression_test(expression, locations, knn=True, n_neighbors=6, d=1,):
    if knn:
        adj = kneighbors_graph(locations, n_neighbors=n_neighbors, mode="connectivity")
        edge_list = np.argwhere(adj==1)
    else:
        _, edge_list, _, _ = create_graph(locations, d=d)
    
    
    all_exp = np.concatenate([expression[:,0][edge_list[:,0]].reshape(-1,1), 
                              expression[:,1][edge_list[:,1]].reshape(-1,1)],
                              axis=1)
    coe_all, f_stat, f_p_value = r_lm(all_exp, cal_f=True)
    return coe_all, f_stat, f_p_value
