import time
import torch
import warnings
import numpy as np
import scanpy as sc
import networkx as nx
from tqdm import tqdm
from scipy import sparse as sps
from .utils.__data_utils import *
from typing import List, Union, Iterable
from matplotlib import pyplot as plt
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from typing import Any, Union, Dict, List
from scipy.spatial import distance_matrix
from .utils.__base import FeaturePropagate
from sklearn.neighbors import kneighbors_graph

class STGraph:

    _settings = None
    _mapping = None
    _additional_notes = None

    def __init__(
            self,
            node_attr: np.ndarray = np.array([]),
            node_coor: np.ndarray = np.array([]),
            node_id: np.ndarray = np.array([]),
            node_stat: np.ndarray = np.array([]),
            section_id: np.ndarray = None,
            mask_stat: np.ndarray = None,
            node_label: np.ndarray = None,
            hv_genes: np.ndarray = None,
            adjacency: sps.spmatrix = sps.csr_matrix((0,0)),
            edge_attr: sps.spmatrix = sps.csr_array((0,0)),
            ) -> None:
        self.__dim_check__(node_attr=node_attr,node_coor=node_coor,node_stat=node_stat,
                        adjacency=adjacency,edge_attr=edge_attr)
        self.node_attr = node_attr
        self.node_coor = node_coor
        self.node_id = node_id
        self.node_stat = node_stat
        self.section_id = section_id
        self.mask_stat = mask_stat
        self.node_label = node_label
        self.hv_genes = hv_genes
        self.adjacency = adjacency
        self.edge_attr = edge_attr
        self.__rearange__()

    @classmethod
    def graphFromAnndata(
        cls,
        adata: sc.AnnData,
        attr_loc: Union[str, List[str]] = None,
        coor_loc: Union[str, List[str]] = ['obsm', 'spatial'],
        label_loc: Union[str, List[str]] = None,
        ## graph construction strategy
        knn: bool = False,
        kdtree: bool = False,
        k: int = 20,
        d: float = None,
        n_adj: int = 4,
        d_eps: float = 1e-6,
        eps: float = 1e-8,
        ## other kwargs
        **kwargs
        ) -> object:

        if attr_loc is None:
            if isinstance(adata.X, np.ndarray):
                node_attr = adata.X
            else:
                node_attr = adata.X.A
        else:
            node_attr = slice_stdata(adata, attr_loc)

        nonzero_columns = np.nonzero(np.any(node_attr, axis=0))[0]
        node_attr = node_attr[:, nonzero_columns]

        if 'highly_variable' in adata.var.columns:
            hv_genes = adata.var['highly_variable']
            hv_genes = hv_genes[nonzero_columns]
            hv_genes = np.where(hv_genes)[0]
        else:
            hv_genes = None

        node_stat = cls.__check_node_stat__(node_attr)
        if node_stat.sum()!=node_stat.shape[0]:
            mask_stat = node_stat.copy()
        else:
            mask_stat = None
        node_id = adata.obs.index.to_numpy()
        if not np.unique(node_id).shape[0] == node_id.shape[0]:
            warnings.warn("'obs_name' not unique. Please run 'adata.obs_names_make_unique' at first for index security.")
        
        node_coor = slice_stdata(adata,coor_loc)
        mapping, adjacency, edge_attr, d = cls.__create_graph__(
            node_coor,
            node_attr,
            knn = knn,
            kdtree = kdtree,
            k = k,
            d = d,
            n_adj = n_adj,
            d_eps = d_eps,
            eps = eps
        )
        cls._mapping = mapping
        cls._settings = dict(
            knn = knn,
            kdtree = kdtree,
            k = k,
            d = d,
            n_adj = n_adj,
            d_eps = d_eps,
            eps = eps
        )

        if label_loc is not None:
            node_label = np.array(slice_stdata(adata, label_loc)).reshape(-1)
        else:
            node_label = None

        return cls(
            node_attr=node_attr, node_coor=node_coor, node_stat=node_stat, node_label=node_label, hv_genes = hv_genes,
            adjacency=adjacency, edge_attr=edge_attr, node_id=node_id, mask_stat=mask_stat)
    
    @classmethod
    def graphFrom3DAnndata(
        cls,
        adata: sc.AnnData,
        ordered_section_name: Iterable,
        attr_loc: Union[str, List[str]] = None,
        coor_loc: Union[str, List[str]] = ['obsm', 'spatial'],
        label_loc: Union[str, List[str]] = None,
        section_id: Union[str, List[str]] = ['obs', 'section_id'],
        section_dist: Iterable = None,
        ## graph construction strategy
        knn: bool = False,
        kdtree: bool = False,
        k: int = 20,
        d: float = None,
        n_adj: int = 4,
        d_eps: float = 1e-6,
        eps: float = 1e-8,
        between_section_k: int = 3,
        ## other kwargs
        **kwargs
        ) -> object:
        
        if attr_loc is None:
            if isinstance(adata.X, np.ndarray):
                node_attr = adata.X
            else:
                node_attr = adata.X.A
        else:
            node_attr = slice_stdata(adata, attr_loc)

        nonzero_columns = np.nonzero(np.any(node_attr, axis=0))[0]
        node_attr = node_attr[:, nonzero_columns]

        if 'highly_variable' in adata.var.columns:
            hv_genes = adata.var['highly_variable']
            hv_genes = hv_genes[nonzero_columns]
            hv_genes = np.where(hv_genes)[0]
        else:
            hv_genes = None

        node_stat = cls.__check_node_stat__(node_attr)
        if node_stat.sum()!=node_stat.shape[0]:
            mask_stat = node_stat.copy()
        else:
            mask_stat = None
        node_id = adata.obs.index.to_numpy()
        if not np.unique(node_id).shape[0] == node_id.shape[0]:
            warnings.warn("'obs_name' not unique. Please run 'adata.obs_names_make_unique' at first for index security.")
        
        node_coor = slice_stdata(adata,coor_loc)
        section_id, section_dist = np.array(slice_stdata(adata,section_id)), np.array(section_dist)
        section_id = np.array([str(item) for item in section_id])
        ordered_section_name = np.array([str(item) for item in ordered_section_name])
        mapping, adjacency, edge_attr, d = cls.__create_3D_graph__(
            node_coor,
            node_attr,
            section_id = section_id,
            section_dist = section_dist,
            ordered_section_name = ordered_section_name,
            knn = knn,
            kdtree = kdtree,
            k = k,
            d = d,
            n_adj = n_adj,
            d_eps = d_eps,
            eps = eps,
            between_section_k=between_section_k
        )
        cls._mapping = mapping
        cls._settings = dict(
            section_dist = section_dist,
            ordered_section_name = ordered_section_name,
            knn = knn,
            kdtree = kdtree,
            k = k,
            d = d,
            n_adj = n_adj,
            d_eps = d_eps,
            eps = eps,
            between_section_k=between_section_k
        )

        if label_loc is not None:
            node_label = np.array(slice_stdata(adata, label_loc)).reshape(-1)
        else:
            node_label = None

        return cls(
            node_attr=node_attr, node_coor=node_coor, node_stat=node_stat, node_label=node_label, hv_genes = hv_genes,
            adjacency=adjacency, edge_attr=edge_attr, node_id=node_id, mask_stat=mask_stat, section_id=section_id
            )

    @property
    def X(self):
        return self.node_attr
    
    @property
    def A(self):
        return self.adjacency

    @property
    def E_list(self):
        row = self.adjacency.tocoo().row
        col = self.adjacency.tocoo().col
        return np.concatenate([row.reshape(-1,1),col.reshape(-1,1)],axis=1)
    
    @property
    def E_attr(self):
        return self.edge_attr.tocoo().data

    @property
    def I(self):
        n = len(self.node_attr)
        return sps.csr_matrix(
            (np.ones(n), (np.arange(n),np.arange(n))),
            shape = (n,n) 
        )
    
    @property
    def I_torch(self):
        n = self.node_attr.shape[0]
        return torch.sparse_coo_tensor(
            indices=torch.cat([torch.arange(n).reshape(1, -1)] * 2, dim=0),
            values=torch.FloatTensor(torch.ones(n)),
            size = (n, n)
        )

    @property
    def Unique_Edges(self):
        n = self.adjacency.shape[0]
        edge_list = []
        for i in range(n):
            row = self.adjacency.getrow(i)
            idx = np.argwhere(row > 0).reshape(-1)
            for j in idx:
                if j > i:
                    edge_list.append([i,j])
        return np.array(edge_list)
    
    @property
    def HVG_Ind(self):
        return self.hv_genes
    
    def pca(
            self, n_components: int = 300, scalling: str = "None", 
            random_state: int = 42, svd_solver: str = 'full', device: str = 'cpu',
            use_hvg: bool = True, in_place: bool = True
        ):
        if use_hvg and self.hv_genes is not None:
            adata_temp = sc.AnnData(X=self.node_attr)[:, self.hv_genes]
            print(f"PCA pre-compression for data onto {n_components}-dim.")
            print(f"Use Top-{len(self.hv_genes)} (out of {self.node_attr.shape[1]}) genes for compression.")
        else:
            adata_temp = sc.AnnData(X=self.node_attr)
            print(f"PCA pre-compression for data, from {self.node_attr.shape[1]} onto {n_components}-dim.",)
        print(f"Scaling data: {scalling}; SVD solver: {svd_solver}; random_state={random_state}.")
        print("Start compression...", end="\t")
        start_time = time.time()
        if in_place:
            self.node_attr = pca_adata(
                adata_temp, np.argwhere(self.node_stat==1).reshape(-1), np.argwhere(self.node_stat==0),
                n_components=n_components, device=device,
                scalling=scalling, svd_solver=svd_solver, random_state=random_state)
        else:
            self.add_additional_notes(
                "node_attr_PCA",
                pca_adata(
                adata_temp, np.argwhere(self.node_stat==1).reshape(-1), np.argwhere(self.node_stat==0),
                n_components=n_components, device=device,
                scalling=scalling, svd_solver=svd_solver, random_state=random_state
                )
                )
        end_time = time.time()
        print(f"Done! Elapsed time: {end_time - start_time:.2f}s.")

        gc.collect()
        torch.cuda.empty_cache()

    def ca(self, n_components: int = 300, svd_solver: str = 'full', random_state = 42, ca_kwargs: dict = {}, in_place: bool = True):
        adata_temp = sc.AnnData(X=self.node_attr)
        print(f"SVD solver: {svd_solver}; random_state={random_state}.")
        print("Start compression...", end="\t")
        start_time = time.time()
        if in_place:
            self.node_attr = ca_adata(
                adata_temp, np.argwhere(self.node_stat==1).reshape(-1), np.argwhere(self.node_stat==0),
                    n_components=n_components, svd_solver=svd_solver, **ca_kwargs 
                )
        else:
            self.add_additional_notes(
                "node_attr_CA",
                ca_adata(
                adata_temp, np.argwhere(self.node_stat==1).reshape(-1), np.argwhere(self.node_stat==0),
                    n_components=n_components, svd_solver=svd_solver, **ca_kwargs 
                ))
        end_time = time.time()
        print(f"Done! Elapsed time: {end_time - start_time:.2f}s.")

    def scvi(self, batch_id: Any=None, max_epochs: int=400, scvi_save_path: str = './', 
            scvi_kwargs: dict = dict(n_hidden=256, n_latent=64, n_layers=2), in_place: bool = True):
        adata_temp = sc.AnnData(X=self.node_attr)
        if batch_id is not None:
            adata_temp.obs[batch_id] = self._additional_notes[batch_id]
        print("SCVI pre-compression for data.",)
        if in_place:
            self.node_attr = scvi_adata(
                adata=adata_temp, pos_idx=np.argwhere(self.node_stat==1).reshape(-1), neg_idx=np.argwhere(self.node_stat==0), 
                scvi_kwargs=scvi_kwargs, batch_id=batch_id, scvi_save_path=scvi_save_path, max_epochs=max_epochs
                ).astype("float32")
        else:
            self.add_additional_notes(
                "node_attr_scvi",
                scvi_adata(
                adata=adata_temp, pos_idx=np.argwhere(self.node_stat==1).reshape(-1), neg_idx=np.argwhere(self.node_stat==0), 
                scvi_kwargs=scvi_kwargs, batch_id=batch_id, scvi_save_path=scvi_save_path, max_epochs=max_epochs
                ).astype("float32")
                )

    def padding(
            self,
            reconstruct_knn: bool = False,
            reconstruct_k: int = None
            ) -> None:
        edge_list = self.Unique_Edges
        new_coor =  0.5 * (self.node_coor[edge_list[:, 0]] + self.node_coor[edge_list[:, 1]])
        new_attr = np.zeros((edge_list.shape[0],self.node_attr.shape[1]))
        self.node_padd = np.array([0]*self.node_attr.shape[0]+[1]*new_coor.shape[0])
        self.node_id = np.concatenate([self.node_id, [f"pad_{ii+1}" for ii in range(new_coor.shape[0])]])
        self.node_attr = np.concatenate([self.node_attr,new_attr],axis=0)
        self.node_coor = np.concatenate([self.node_coor,new_coor],axis=0)
        self.node_stat = np.concatenate([self.node_stat,np.zeros(edge_list.shape[0])])
        if self.mask_stat is not None:
            self.mask_stat = np.concatenate([self.mask_stat, np.ones(new_coor.shape[0])])
        if reconstruct_knn:
            self._settings['knn'] = reconstruct_knn
            self._settings['k'] = reconstruct_k
        elif self._settings['d'] is None and self._settings['knn'] == False:
            self._settings['knn'] = True
            self._settings['k'] = 18
        else:
            self._settings['d'] /= 2

        self._mapping, self.adjacency, self.edge_attr, _ = self.__create_graph__(
            self.node_coor,
            self.node_attr,
            **self._settings
        )

        self.__dim_check__(
            node_attr=self.node_attr,node_coor=self.node_coor,node_stat=self.node_stat,
            adjacency=self.adjacency,edge_attr=self.edge_attr)

    def padding3D(self, threshold: float = None):
        r'''
            This method doesn't support graph reconstruction at this state
        '''
        ordered_section_name = self._settings['ordered_section_name']
        section_dist = self._settings['section_dist']
        n_sections = len(ordered_section_name)
        new_ordered_section, section_to_drop = [], []
        loc2section = dict()
        for i in range(n_sections-1):
            section_this = ordered_section_name[i]; section_next = ordered_section_name[i+1]
            section_add_between = str(section_this)+"-"+str(section_next)
            pos_this = self.node_coor[self.section_id==section_this,2][0]
            pos_next = self.node_coor[self.section_id==section_next,2][0]
            loc2section[pos_this] = str(section_this)
            loc2section[(pos_this+pos_next)/2] = section_add_between
            new_ordered_section.append(section_this)  
            if (threshold is None) or (section_dist[i] > threshold):
                new_ordered_section.append(section_add_between)
            elif section_dist[i] <= threshold:
                section_to_drop.append(section_add_between)
            else:
                raise ValueError(f'Unpaired "section" distance and "threshold!"')
        loc2section[self.node_coor[self.section_id==ordered_section_name[-1],2][0]] = str(ordered_section_name[-1])
        new_ordered_section.append(ordered_section_name[-1])
        self._settings['ordered_section_name'] = new_ordered_section
        
        # start padding
        n_nodes = self.node_attr.shape[0]
        edge_list = self.Unique_Edges
        new_coor =  0.5 * (self.node_coor[edge_list[:, 0]] + self.node_coor[edge_list[:, 1]])
        new_attr = np.zeros((edge_list.shape[0],self.node_attr.shape[1]))
        new_section_id = np.array([loc2section[loc] for loc in new_coor[:,2]])
        
        # get drop idx
        keep_idx = np.ones(new_coor.shape[0])
        for section in section_to_drop:
            keep_idx[np.argwhere(new_section_id==section)] = 0
        keep_idx = np.concatenate([np.ones(self.node_coor.shape[0]),keep_idx],axis=0)

        self.node_padd = np.array([0]*self.node_attr.shape[0]+[1]*new_coor.shape[0])
        self.node_id = np.concatenate([self.node_id, [f"pad_{ii+1}" for ii in range(new_coor.shape[0])]])
        self.node_attr = np.concatenate([self.node_attr,new_attr],axis=0)
        self.node_coor = np.concatenate([self.node_coor,new_coor],axis=0)
        self.node_stat = np.concatenate([self.node_stat,np.zeros(edge_list.shape[0])])
        self.section_id = np.concatenate([self.section_id,new_section_id],axis=0)
        if self.mask_stat is not None:
            self.mask_stat = np.concatenate([self.mask_stat, np.ones(new_coor.shape[0])])

        new_edge_list, new_edge_attr = [], []
        for i in tqdm(range(edge_list.shape[0]), "Padding:"):
            node_left, node_right = edge_list[i]
            new_edge_list.extend([[node_left,n_nodes],[n_nodes,node_right]])
            new_edge_list.extend([[n_nodes,node_left],[node_right,n_nodes]])
            d = np.sqrt(((self.node_coor[node_left] - self.node_coor[node_right])**2).sum()) / 2
            new_edge_attr.extend([d,d,d,d])
            n_nodes += 1

        edge_list = np.concatenate([self.E_list,new_edge_list],axis=0)
        edge_attr = np.concatenate([self.E_attr,new_edge_attr],axis=0)
        self.adjacency = sps.csr_array(
            (np.ones(edge_list.shape[0]),(edge_list[:,0],edge_list[:,1])),
            shape = (self.node_attr.shape[0], self.node_attr.shape[0])
            )
        self.edge_attr = sps.csr_array(
            (edge_attr,(edge_list[:,0],edge_list[:,1])),
            shape = (self.node_attr.shape[0], self.node_attr.shape[0])
        )

        self.__drop__(np.argwhere(keep_idx==0))
        self.__dim_check__(node_attr=self.node_attr,node_coor=self.node_coor,node_stat=self.node_stat,
                        adjacency=self.adjacency,edge_attr=self.edge_attr,section_id=self.section_id)
        self.__rearange__()
           
    def mask_nodes(
            self,
            mask_idx: np.ndarray = None,
            mask_vert_x: np.ndarray=None,
            mask_vert_y: np.ndarray=None
        ) -> None:
        if mask_idx is None:
            assert len(mask_vert_x) == len(mask_vert_y)
            mask_idx = []
            n_mask_vert = len(mask_vert_x)
            for k in tqdm(range(len(self.node_coor)),'Creating mask indexes'):
                vert_x, vert_y = self.node_coor[k,0], self.node_coor[k,1]
                # check whether the vert is in polygon
                i, j, c = 0, n_mask_vert-1, False
                while i < n_mask_vert:
                    if  ((mask_vert_y[i]>vert_y) != (mask_vert_y[j]>vert_y)) and \
                        (vert_x <= (mask_vert_x[j]-mask_vert_x[i]) * (vert_y-mask_vert_y[i]) / (mask_vert_y[j]-mask_vert_y[i]) + mask_vert_x[i]):
                        c = not c
                    j = i
                    i += 1
                if c:
                    mask_idx.append(k)
        self.node_stat[mask_idx] = 0
        if self.mask_stat is not None:
            self.mask_stat[mask_idx] = 0
        else:
            self.mask_stat = np.ones(self.node_stat.shape[0])
            self.mask_stat[mask_idx] = 0
        self.__rearange__()

    def remove_duplicates(self, r=0.2):
        idx = np.ones(self.node_coor.shape[0])
        d = self._settings['d'] * r

        with tqdm(
            total=self.node_coor.shape[0],
            desc="Detecting duplicates",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for i in range(self.node_coor.shape[0]):
                dist = np.sqrt(((self.node_coor-self.node_coor[i])**2).sum(axis=1))
                dist[i] = d+1
                candidates = np.argwhere(dist<=d)
                for j in candidates:
                    if self.node_stat[i] == 0 and self.node_stat[j]==1:
                        idx[i] = 0
                    elif self.node_stat[j] == 0 and self.node_stat[i]==1:
                        idx[j] = 0
                    elif self.node_stat[i] == 1 and self.node_stat[j] == 1:
                        pass
                    else:
                        if j > i:
                            idx[j] = 0
                        else:
                            idx[i] = 0
                pbar.update(1)
        self.__drop__(np.argwhere(idx==0))

        if self.node_coor.shape[1] == 2:
            self._mapping, self.adjacency, self.edge_attr, _ = self.__create_graph__(
                self.node_coor,
                self.node_attr,
                **self._settings
            )
        else:
            _, self.adjacency, self.edge_attr, _ = self.__create_3D_graph__(
            self.node_coor,
            self.node_attr,
            section_id=self.section_id,
            **self._settings
        )
        self.__dim_check__(node_attr=self.node_attr,node_coor=self.node_coor,node_stat=self.node_stat,
                        adjacency=self.adjacency,edge_attr=self.edge_attr, section_id=self.section_id)
        self.__rearange__()

    def feature_propagate(
        self,
        device: str = "cuda", 
        max_iter: int = 300,
        reproducable: bool = True
    ) -> torch.FloatTensor:
        node_embd = torch.FloatTensor(self.node_attr[self.node_stat == 1])
        if reproducable:
            edge_idx = torch.LongTensor(self.E_list[self._mapping].T)
            edge_attr = 1 / torch.FloatTensor(self.E_attr[self._mapping])
        else:
            edge_idx = torch.LongTensor(self.E_list.T)
            edge_attr = torch.FloatTensor(1 / self.E_attr)
        value_idx = torch.LongTensor(np.argwhere(self.node_stat == 1).reshape(-1))
        n_nodes = self.node_attr.shape[0]

        node_fp = FeaturePropagate(node_embd, edge_idx, edge_attr, value_idx, n_nodes,
                                   device=device, max_iter=max_iter)
        if not device == "cpu":
            node_fp = node_fp.cpu().detach()
        else:
            node_fp = node_fp.detach()

        return node_fp
    
    def add_additional_notes(self, key: Any, value: np.ndarray):
        if value.shape[0] != self.node_attr.shape[0]:
            raise ValueError(f"Invalid value dim: Expect value to have size ({self.node_attr.shape[0]},) but with size {value.shape}")
        if self._additional_notes is None:
            self._additional_notes = {key: value}
        else:
            self._additional_notes[key] = value

    def get_additinal_notes(self, key: Any):
        return self._additional_notes[key]

    def topyg(self, pyg_edge_attr:bool = False, reproducable:bool = True, use_device: str='cuda'):
        edge_index = torch.LongTensor(self.E_list).T
        x_id = self.node_id
        if pyg_edge_attr:
            edge_attr = 1 / torch.FloatTensor(self.E_attr).reshape(-1,1)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1], value=edge_attr,
                sparse_sizes=(len(self.node_stat),len(self.node_stat))
                ).t()
        else:
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(len(self.node_stat),len(self.node_stat))
                ).t()
        params = dict(
            x = torch.FloatTensor(self.X),
            xfp = self.feature_propagate(reproducable=reproducable, device=use_device),
            adj_t = adj_t,
            inv_dist = 1 / torch.FloatTensor(self.E_attr),
            # I = self.I_torch,
            value_idx = torch.LongTensor(np.argwhere(self.node_stat==1).reshape(-1)),
            infer_idx = torch.LongTensor(np.argwhere(self.node_stat==0).reshape(-1)),
            coord = torch.FloatTensor(self.node_coor),
            x_id = x_id,
            mask_idx = torch.LongTensor(np.argwhere(self.mask_stat==0).reshape(-1)),
        )
        return Data(**params)
    
    def visualize(
        self,
        save: str = None,
        node_size: int = 5,
        fig_kwargs: dict = dict(figsize=(6,6),dpi=100),
        kwargs: dict = {}
        ):
        '''
            Visualize the stgraph  
        '''
        net = nx.Graph()
        net.add_nodes_from(np.arange(self.node_attr.shape[0]))
        net.add_edges_from(self.E_list)
        plt.figure(**fig_kwargs)
        nx.draw_networkx(net, node_color=self.node_stat,pos=self.node_coor, node_size=node_size,**kwargs,
                         with_labels=False)
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()

    def __dim_check__(self, **args) -> None:
        local_var = args
        dim1 = []
        for val in local_var.values():
            if hasattr(val, 'shape'):
                dim1.append(val.shape[0])
            else:
                pass
        # dim1 = [i.shape[0] for i in local_var.values()]
        if len(set(dim1)) != 1:
            raise ValueError(f"Inproperate dimensions of graph conponents: {dim1} for {local_var.keys()} respectively")        

    def __rearange__(self):
        '''
            Rearange the whole graph components by node_stats (positive nodes in before)
            This is for the convenience of deep learning modules
        '''
        pos_idx, neg_idx = np.argwhere(self.node_stat == 1), np.argwhere(self.node_stat == 0)
        idx = np.concatenate([pos_idx,neg_idx]).reshape(-1)

        self.node_attr = self.node_attr[idx]
        self.node_coor = self.node_coor[idx]
        self.adjacency = self.adjacency[idx][:,idx]
        self.edge_attr = self.edge_attr[idx][:,idx]
        self.node_stat = self.node_stat[idx]
        self.node_id = self.node_id[idx]
        if self.mask_stat is not None:
            self.mask_stat = self.mask_stat[idx]
        if self.node_label is not None:
            self.node_label = self.node_label[idx]
        if self.section_id is not None:
            self.section_id = self.section_id[idx]
    
    @staticmethod
    def __create_graph__(
        node_coor: np.ndarray,
        node_attr: np.ndarray,
        knn: bool = False,
        kdtree: bool = False,
        k: int = 20,
        d: float = None,
        n_adj: int = 4,
        d_eps: float = 1e-6,
        eps: float = 1e-8,
    ):
        mapping, edge_list, edge_attr, d = create_graph(
            node_coor,
            knn = knn,
            kdtree = kdtree,
            k = k,
            d = d,
            n_adj = n_adj,
            d_eps = d_eps,
            eps = eps
        )
        adjacency = sps.csr_array(
            (np.ones(edge_list.shape[0]),(edge_list[:,0],edge_list[:,1])),
            shape = (node_attr.shape[0],node_attr.shape[0])
            )
        edge_attr = sps.csr_array(
            (edge_attr,(edge_list[:,0],edge_list[:,1])),
            shape = (node_attr.shape[0],node_attr.shape[0])
        )
        
        return mapping, adjacency, edge_attr, d
    
    @staticmethod
    def __create_3D_graph__(
        node_coor: np.ndarray,
        node_attr: np.ndarray,
        section_id: np.ndarray,
        ordered_section_name: np.ndarray,
        section_dist: np.ndarray = None,
        knn: bool = False,
        kdtree: bool = False,
        k: int = 20,
        d: float = None,
        n_adj: int = 4,
        d_eps: float = 1e-6,
        eps: float = 1e-8,
        between_section_k: int = 3
    ):
        n_sections = len(ordered_section_name)
        n_nodes_count = 0
        edge_lists, edge_attrs, ds = [], [], []
        # construct within-section graph
        for i in range(n_sections):
            section = ordered_section_name[i]
            section_idx = np.argwhere(section_id==section).reshape(-1)
            section_coor = node_coor[section_idx]
            print(f"Section {i}: In total {section_coor.shape[0]} cells, processing...")
            _, edge_list, edge_attr, d = create_graph(
                section_coor[:,:2],
                knn = knn,
                kdtree = kdtree,
                k = k,
                d = d,
                n_adj = n_adj,
                d_eps = d_eps,
                eps = eps
            )
            edge_list += n_nodes_count
            edge_lists.append(edge_list); edge_attrs.append(edge_attr); ds.append(d)
            n_nodes_count += section_coor.shape[0]
            
        # construct between-section graph
        n_nodes_count_this, n_nodes_count_next = 0, 0
        for j in range(n_sections-1):
            section_this = ordered_section_name[j]
            section_next = ordered_section_name[j+1]
            print(f'Between section graph construction: {section_this}-{section_next}, processing...')
            section_this_idx = np.argwhere(section_id==section_this).reshape(-1)
            section_next_idx = np.argwhere(section_id==section_next).reshape(-1)
            section_this_coor = node_coor[section_this_idx]
            section_next_coor = node_coor[section_next_idx]
            n_nodes_count_next += section_this_coor.shape[0]
            dist_m = distance_matrix(section_this_coor, section_next_coor)
            if section_next_coor.shape[0] > section_this_coor.shape[0]:
                kth_smallest = np.argpartition(dist_m, between_section_k, axis=0)[:between_section_k]
                edge_list, edge_attr = [], []
                for j in range(section_next_coor.shape[0]):
                    for i in kth_smallest[:,j]:
                        edge_list.append([i,j]); edge_attr.append(dist_m[i,j])
            else:
                kth_smallest = np.argpartition(dist_m, between_section_k, axis=1)[:,:between_section_k]
                edge_list, edge_attr = [], []
                for i in range(section_this_coor.shape[0]):
                    for j in kth_smallest[i]:
                        edge_list.append([i,j]); edge_attr.append(dist_m[i,j])
            edge_list, edge_attr = np.array(edge_list), np.array(edge_attr)
            edge_list[:,0] += n_nodes_count_this
            edge_list[:,1] += n_nodes_count_next
            edge_list_T = edge_list[:,[1,0]].copy()
            edge_lists.extend([edge_list,edge_list_T]); edge_attrs.extend([edge_attr,edge_attr])
            n_nodes_count_this += section_this_coor.shape[0]

        edge_list = np.concatenate(edge_lists,axis=0)
        edge_attr = np.concatenate(edge_attrs,axis=0)
        max_d = max(ds)
        adjacency = sps.csr_array(
            (np.ones(edge_list.shape[0]),(edge_list[:,0],edge_list[:,1])),
            shape = (node_attr.shape[0],node_attr.shape[0])
            )
        edge_attr = sps.csr_array(
            (edge_attr,(edge_list[:,0],edge_list[:,1])),
            shape = (node_attr.shape[0],node_attr.shape[0])
        )
        print(f'Everything is done!')
        return _, adjacency, edge_attr, max_d

    @staticmethod
    def __check_node_stat__(node_attr):
        nan_rows = np.argwhere(np.isnan(node_attr).any(axis=1))
        inf_rows = np.argwhere(np.isinf(node_attr).any(axis=1))
        zero_rows = np.argwhere(node_attr.sum(axis=1)==0)
        rows =  np.unique(np.concatenate([nan_rows,inf_rows,zero_rows]).reshape(-1))
        node_stat = np.ones(node_attr.shape[0])
        node_stat[rows] = 0
        return node_stat.reshape(-1)
    
    def __drop__(self,index: np.ndarray):

        idx = np.array([True]*self.node_attr.shape[0])
        idx[index] = False        

        self.node_attr = self.node_attr[idx]
        self.node_coor = self.node_coor[idx]
        self.adjacency = self.adjacency[idx][:,idx]
        self.edge_attr = self.edge_attr[idx][:,idx]
        self.node_stat = self.node_stat[idx]
        self.node_id = self.node_id[idx]
        if self.mask_stat is not None:
            self.mask_stat = self.mask_stat[idx]
        if self.node_label is not None:
            self.node_label = self.node_label[idx]
        if self.section_id is not None:
            self.section_id = self.section_id[idx]
