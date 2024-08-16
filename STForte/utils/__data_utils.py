import os
import gc
import torch
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm.auto import tqdm
from torchvision import models
from typing import Dict, List, Union, Any
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import TensorDataset, DataLoader

def slice_stdata(
        stdata:sc.AnnData,
        slice: List[str]
        ):
        assert slice[0] in ['obs','var','obsm','uns']

        if slice[0] in ['obs','var']:
            tar = stdata.obs if slice[0]=='obs' else stdata.var
            if len(slice) == 2:
                return np.array(tar[slice[1]])
            elif len(slice) > 2:
                if isinstance(tar,pd.DataFrame):
                    return np.array(tar[[*slice[1:]]])
                elif isinstance(tar,Dict):
                    return np.concatenate(
                        [np.array(tar[slice[i+1]]) for i in range(len(slice)-1)],
                        axis=1
                    )
            else:
                raise ValueError('Invalide keys for obs/var extraction')
        elif slice[0] == 'obsm':
            return np.array(stdata.obsm[slice[1]])
        elif slice[0] == 'uns':
            tar=stdata.uns
            for key in slice[1:]:
                tar = tar[key]
            return np.array(tar)


def get_pos_and_neg_idx(node_attr:np.ndarray):
    pos_idx = [i for i in range(len(node_attr)) if not np.all(node_attr[i]==0) and not np.all(np.isnan(node_attr[i]))]
    neg_idx = list(set(range(len(node_attr))).difference(set(pos_idx)))
    return pos_idx, neg_idx

def create_graph(
        node_coor: np.ndarray,
        knn: bool = False,
        kdtree: bool = False,
        k: int = 20,
        d: float = None,
        n_adj: int = 4,
        d_eps: float = 1e-6,
        eps: float = 1e-8,
    ):
    if knn:
        edge_list, edge_attr = knn_graph(node_coor,k,kdtree=kdtree)
    else:
        if d is None:
            d = infer_d(node_coor, n_adj, d_eps)
            print(f'Infered d value: {d}')
        else:
            d = d + eps
        edge_list, edge_attr = knn_graph(node_coor, d=d)
    idx = np.lexsort((edge_list[:,1], edge_list[:,0])).reshape(-1,1)
    map_dic = {i:j for i,j in np.concatenate([idx,np.arange(edge_list.shape[0]).reshape(-1,1)],axis=1)}
    mapping = [map_dic[i] for i in np.arange(edge_list.shape[0])]
    mapping = np.array(mapping)
    
    return mapping, edge_list, edge_attr, d

def pca_adata(
    adata: sc.AnnData,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_components: int,
    scalling: Union[str,bool],
    svd_solver: str,
    random_state: int,
    target_sum: int=None,
    device: str = 'cpu'
    ):
    sc.pp.normalize_total(adata,target_sum=target_sum)
    if scalling == "z_score":
        adata.X = (adata.X - adata.X.mean(0)) / adata.X.std(0)
    elif scalling == "centering":
        adata.X = adata.X - adata.X.mean(0)
    elif scalling == "None" or scalling == False:
        pass
    else:
        ValueError(f"Invaild scalling method {scalling} for PCA! Select method from ['z_score','centering'].")
    if len(neg_idx) == 0:
        if svd_solver == 'torch':
            return pca(adata.X, n_components=n_components, device=device)
        else:
            adata = sc.pp.pca(adata, n_comps=n_components, copy=True, svd_solver=svd_solver, random_state=random_state)
            return np.array(adata.obsm['X_pca'])
    else:
        n_nodes = len(pos_idx) + len(neg_idx)
        node_attr = np.zeros((n_nodes,n_components))
        adata_mask = adata[pos_idx].copy()
        if svd_solver == 'torch':
            node_attr[pos_idx] = pca(adata_mask.X,n_components=n_components,device=device)
        else:
            adata_mask = sc.pp.pca(adata_mask, n_comps=n_components, copy=True, svd_solver=svd_solver, random_state=random_state)
            node_attr[pos_idx] = np.array(adata_mask.obsm['X_pca'])
        node_attr[neg_idx] = np.nan
        return node_attr
    

def ca_adata(adata: sc.AnnData,
             pos_idx: np.ndarray,
             neg_idx: np.ndarray,
             n_components: int = None,
             svd_solver: str = "randomized",
             random_state: int = 42,
             svd_kwargs: dict = {}):
    if isinstance(adata.X, csr_matrix):
        X = adata.X.A
    else:
        X = adata.X
    if len(neg_idx) == 0:
        ca_embedd = corr_analysis(X, n_components, svd_solver, random_state=random_state, **svd_kwargs)
        return ca_embedd
    else:
        X_pos = X[pos_idx]
        # adata_mask = adata[pos_idx].copy()
        # adata_mask = sc.pp.pca(adata_mask,n_components,copy=True)
        ca_embedd = corr_analysis(X_pos, n_components, svd_solver, random_state=random_state, **svd_kwargs)
        n_nodes = len(pos_idx) + len(neg_idx)
        node_attr = np.zeros((n_nodes,n_components))
        # node_attr[pos_idx] = np.array(adata_mask.obsm['X_pca'])
        node_attr[pos_idx] = ca_embedd
        node_attr[neg_idx] = np.nan
        return node_attr


def scvi_adata(
    adata: sc.AnnData,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    scvi_kwargs: Dict,
    max_epochs:int=400,
    batch_id: Any = None,
    scvi_save_path: str = None
    ):
    import scvi
    if len(neg_idx) == 0:
        if batch_id is not None:
            scvi.model.SCVI.setup_anndata(adata, batch_key=batch_id)
        scvi.model.SCVI.setup_anndata(adata)
        print(scvi_kwargs)
        vae = scvi.model.SCVI(adata, **scvi_kwargs)
        vae.train()
        if scvi_save_path is not None:
            if not os.path.exists(scvi_save_path):
                os.makedirs(scvi_save_path)
        vae.save(scvi_save_path, save_anndata=True, overwrite=True)
        return np.array(vae.get_latent_representation())
    else:
        adata_mask = adata[pos_idx].copy()
        scvi.model.SCVI.setup_anndata(adata_mask)
        print(scvi_kwargs)
        vae = scvi.model.SCVI(adata_mask, **scvi_kwargs)
        vae.train(max_epochs)
        if scvi_save_path is not None:
            if not os.path.exists(scvi_save_path):
                os.makedirs(scvi_save_path)
            vae.save(scvi_save_path, save_anndata=True, overwrite=True)
        n_nodes = len(pos_idx) + len(neg_idx)
        pos_scvi = np.array(vae.get_latent_representation())
        node_attr = np.zeros((n_nodes,pos_scvi.shape[1]))
        node_attr[pos_idx] = pos_scvi
        node_attr[neg_idx] = np.nan
        return node_attr

def knn_graph(
    node_coor: np.ndarray,
    k: int = None, 
    d: float = None,
    kdtree: bool = False,
    ):
    ''' 
        construct graph with distance matrix
    '''
    edge_attr = []
    edge_list = []
    if d is None:
        if kdtree:
            print("KDTree knn innitialize.")
            tree = KDTree(node_coor)
            dist, ind = tree.query(node_coor, k=k + 1)
            edge_list = np.concatenate([np.array([[nn] * k for nn in range(len(node_coor))]).reshape(1, -1), ind[:, 1:].reshape(1, -1)], axis=0).T
            edge_attr = dist[:, 1:].reshape(-1)
        else:
            for i in tqdm(range(len(node_coor)), 'brute-force knn initialize'):
                dist = np.sqrt(((node_coor - node_coor[i]) ** 2).sum(axis=1))
                idx = np.argsort(dist)[1:(k+1)]
                d_nei = dist[idx]
                for j in range(len(idx)):
                    edge = (i,int(idx[j]))
                    edge_list.append(edge)
                    edge_attr.append(d_nei[j])
    else:
        for i in tqdm(range(len(node_coor)),'d-based initialize'):
            dist = np.sqrt(((node_coor-node_coor[i])**2).sum(axis=1))
            idx = np.argwhere(np.logical_and(dist <= d, dist != 0))
            d_nei = dist[idx]
            for j in range(len(idx)):
                edge = (i,int(idx[j]))
                edge_list.append(edge)
                edge_attr.append(d_nei[j])
    edge_list, edge_attr = np.array(edge_list).reshape((-1,2)), np.array(edge_attr).reshape(-1)
    return edge_list, edge_attr

def infer_d(
        node_coor: np.ndarray,
        k: int,
        eps_infer: float
        ):
        rng = np.random.default_rng(42)
        
        dif_ = []
        x_max = []
        
        for s in range(100):
            idx = rng.choice(np.arange(len(node_coor)))
            x = np.sqrt(((node_coor - node_coor[idx]) ** 2).sum(axis=1))
            x = np.sort(x)[1:k+1]
            dif = np.diff(x)
            dif_.append(dif)
            if np.all(dif<=eps_infer):
                x_max.append(x.max())
            
            if len(x_max) == 20:
                return max(x_max)
            
            if s==99:
                print('Fail to infer d within 100 times samplling! Please check input parameters and data! ')
            
            # seed += 1

        dif_ = np.array(dif_).reshape(-1,k-1)
        print(f'The average distance between nodes: {dif_.mean(axis=0)}')
        raise RuntimeError('Fail to infer d!')

def tilling(
    img: np.ndarray,
    spot_loc: np.ndarray,
    crop_size: int=20,
    target_size: int=299,
    save_dir: str=None
    ):
    ''' tilling the morphology figure '''

    # --- control img type --- #
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    img_pillow = Image.fromarray(img)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")
    
    _,_,b1,b2 = img_pillow.getbbox()
    spot_loc[:,0] = b1 * (spot_loc[:,0]-spot_loc[:,0].min()) / (spot_loc[:,0].max()-spot_loc[:,0].min())
    spot_loc[:,1] = b2 * (spot_loc[:,1]-spot_loc[:,1].min()) / (spot_loc[:,1].max()-spot_loc[:,1].min())

    tile_list = []

    with tqdm(
    total=len(spot_loc),
    desc="Tiling image",
    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
) as pbar:
        for r,c in spot_loc:
            r_down  = r - crop_size / 2
            r_up    = r + crop_size / 2
            c_left  = c - crop_size / 2
            c_right = c + crop_size / 2
            tile = img_pillow.crop(
                (c_left, r_down, c_right, r_up)
            )
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
            tile.resize((target_size, target_size))
            if save_dir is not None:
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                tile_name = str(r) + "-" + str(c) + "-" + str(crop_size)
                out_tile = save_dir + tile_name + ".jpeg"
                tile.save(out_tile, "JPEG")
            tile_list.append(np.array(tile))
            pbar.update(1)

    return np.array(tile_list)


def extract_features(
    tiling_result,
    model: str='Resnet50',
    use_pca: bool=True,
    n_components: int=300
    ):
    ''' extracting morphology fearures '''
    data_loader = DataLoader(TensorDataset(torch.FloatTensor(tiling_result)), batch_size=256)
    if model == 'Resnet50':
        model = models.resnet50()
        model.load_state_dict(torch.load('TOASTModel/utils/pretrained_models/resnet50-19c8e357.pth'))
        model.eval()
    else:
        raise ValueError('Other pretrained model except resnet50 is currentlt not available')

    features = []
    with tqdm(
    total=len(iter(data_loader)),
    desc="Extracting features",
    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
) as pbar:
        for _batch in iter(data_loader):
            _feature = model(_batch[0])
            for i in _feature:
                features.append(i.detach().numpy())
            pbar.update(1)
    features=np.array(features)

    # --- apply pca --- #
    if use_pca:
        if n_components >= features.shape[1]:
            raise ValueError('Components of PCA should be less than the number of features (1000)!')
        _pca = PCA(n_components=n_components)
        features = _pca.fit_transform(features)

    return features


def corr_analysis(X: np.ndarray,
                  n_components: int = 50,
                  svd_solver: str = "randomized",
                  svd_kwargs: dict = {},
                  all_return: bool = False,
                  random_state: int = 42,
                  eps = 1e-6):
    # CA preprocessing
    N = X.sum()
    X = X / N
    col_sum, row_sum = X.sum(axis=0)+eps, X.sum(axis=1)+eps
    expected = np.outer(col_sum,row_sum).T
    r = (X - expected) / np.sqrt(expected)

    # SVD
    if n_components > min(X.shape):
        if svd_solver == "randomized":
            n_components = min(X.shape)
        else:
            n_components = min(X.shape) - 1
        warnings.warn(f'n_compoenents exceed maximum value, reset it to {min(X.shape)}.')
    if svd_solver == "randomized":
        U, S, Vt = randomized_svd(r, n_components=n_components, random_state=random_state, **svd_kwargs)
    else:
        U, S, Vt = svd(r, **svd_kwargs)
        U = U[:,:n_components]
        Vt = Vt[:n_components]
        S = S[:n_components]

    
    U = U / np.sqrt(row_sum).reshape(-1, 1)
    Vt = Vt / np.sqrt(col_sum).reshape(-1, 1).T

    U = U * S
    Vt = Vt * S.reshape(-1, 1)
    if all_return:
        return U, S, Vt
    else:
        return U
    
def pca(X: np.ndarray, n_components: int, device: str = "cpu"):

    X = torch.FloatTensor(X)

    #Step-1
    X_meaned = X - X.mean(dim=0)
    X_copy = torch.clone(X_meaned)

    #Step-2
    cov_mat = torch.cov(X_meaned.T)
    cov_mat = cov_mat.to(device)
    
    #Step-3
    eigen_values , eigen_vectors = torch.linalg.eigh(cov_mat)
    eigen_values , eigen_vectors = eigen_values.to('cpu') , eigen_vectors.to('cpu')
    
    #Step-4
    sorted_index = torch.argsort(eigen_values,descending=True)
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,:n_components]
    
    #Step-6
    X_reduced = torch.matmul(eigenvector_subset.T, X_copy.T).T

    return X_reduced.detach().numpy()