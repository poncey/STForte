
def mclust_R(adata, num_cluster, obs_add='mclust', modelNames='EEE', used_obsm='Z_COMB', random_seed=42):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import numpy as np
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames, verbose=False)
    mclust_res = np.array(res[-2])

    adata.obs[obs_add] = mclust_res
    adata.obs[obs_add] = adata.obs[obs_add].astype('int')
    adata.obs[obs_add] = adata.obs[obs_add].astype('category')
    # Remove redundant (empty) NA category
    if adata.obs[obs_add].isna().sum() == 0:
        adata = adata[~adata.obs[obs_add].isna()]
    return adata
