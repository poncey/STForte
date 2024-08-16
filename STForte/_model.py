import os
import time
import torch
import pickle
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from ._data import STGraph
from torch_geometric import data
from typing import Dict, List, Union
from anndata._core.anndata import AnnData
from .utils.__module import STForteModule
from .utils.__base import FeaturePropagate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans, MiniBatchKMeans
torch.set_float32_matmul_precision('high')
cudnn.deterministic = True
cudnn.benchmark = False

class STForteModel:
    '''
        STForte: A model works for enhanced spatial transcriptomics analysis.

            The implementation here integrated the model structure and the data structure

            Model structure:
                STForte
    '''

    def __init__(self,
                 adata: AnnData,
                 gdata: data.Data = None,
                 distance: float = None,
                 coord=None,
                 # data utilizing configurations
                 padding: bool = True,
                 padding_kwargs: dict = dict(reconstruct_knn=True,reconstruct_k=18),
                 data_kwargs: dict = {},
                 strategy: str = "pca",
                 n_pc: int = 300,
                 scvi_save_path = None,
                 scvi_kwargs: dict = {},
                 # training configurations
                 epochs: int = 450,
                 gpus: int = 1, random_seed: int = 42,
                 # clustering configurations
                 output_dir: str = "STForteOutput",
                 cluster_kwargs: dict = {},
                 module_kwargs: dict = {}
                 ):

        self.hyper_params = locals().copy()
        del self.hyper_params['self']

        # gene/spot identifiers
        self.gene_identifier = adata.var_names.to_series()
        self.spot_identifier = adata.obs_names.to_series()

        # training configurations
        self.seed = random_seed
        self.gpus = gpus
        self.epochs = epochs
        self.output_dir = output_dir
        self.expand_spots = padding
        # clustering configurations

        # working directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Data input
        if strategy == 'count_mse':
            data_kwargs['reduction'] = 'count'
        elif strategy == "pca":
            data_kwargs['reduction'] = "pca"
            data_kwargs['n_components'] = n_pc
        elif strategy == "scvi":
            data_kwargs['reduction'] = "scvi"
            data_kwargs['scvi_kwargs'] = scvi_kwargs
        elif strategy == "ca":
            data_kwargs['reduction'] = "ca"
            data_kwargs["n_components"] = n_pc
        else:
            raise ValueError("'strategy' should be one of the string in ['count_nb', 'count_mse', 'pca', 'scvi'].")

        data_kwargs['scvi_save_path'] = scvi_save_path
        if gdata is None:
            self.stgraph = STGraph.graphFromAnndata(
                adata=adata, coor_loc=coord, d=distance, 
                **data_kwargs
            )
            if padding:
                self.stgraph.padding(**padding_kwargs)
            self.gdata = self.stgraph.topyg()
        else:
            self.stgraph = None
            self.gdata = gdata
        # model and training initialization
        self.module = STForteModule(self.gdata.x.shape[1], **module_kwargs)

    @classmethod
    def __load_from_saved_model__(
        cls,
        dir: str=None,
        gdata_dir: str=None,
        model_dir: str=None,
        hyper_dir: str=None
    ):
        '''
            loading model from specified directory
            1. Specipy the folder of saved model
                -- STForte_model_{time} <-
                 | -- ......
            2. Specipy all the parameter files
                gdata_dir: [str] file dir for model gdata
                model_dir: [str] file dir for model parameters
                hyper_dir: [str] file dir for model hyperparameters

            this initialization support passing both model foder dir and dirs of specialized model parts. In 
            this case, specialized parts of model will be first extracted from the given dir and the remaining
            ungiven parts will be extracted from the model folder.
        '''
        if dir is None:
            assert gdata_dir is str and model_dir is str and hyper_dir is str

        if dir[-1] != '/':
            dir += '/'

        if hyper_dir is not None:
            with open(hyper_dir,'rb') as f:
                hyper_params = pickle.load(f)
                f.close()
        else:
            with open(dir + 'hyper_params.pkl','rb') as f:
                hyper_params = pickle.load(f)
                f.close()

        if gdata_dir is not None:
            with open(gdata_dir,'rb') as f:
                gdata = pickle.load(f)
                f.close()
        else:
            with open(dir + 'gdata.pkl','rb') as f:
                gdata = pickle.load(f)
                f.close()

        hyper_params['gdata'] = gdata

        model = cls(**hyper_params)

        if model_dir is not None:
            model.load_from_state_dict(model_dir)
        else:
            model.load_from_state_dict(dir + 'model.pt')

        return model

    def fit(self, shuffle: bool = False):
        """fit the STForteModule with custom training arguments

        Args:
            shuffle (bool, optional): whether to shuffle training set. Defaults to
                False.
            load_best_epoch (bool, optional): whether to load the best epoch after
                training. Defaults to True.
        """
        dataloader = DataLoader([self.gdata], batch_size=1,
                            shuffle=shuffle,
                            collate_fn=lambda xs:xs[0])
        pl.seed_everything(self.seed)
        self.module.train()
        trainer = pl.Trainer(max_epochs=self.epochs, log_every_n_steps=1,
                                default_root_dir=self.output_dir, 
                                gpus=self.gpus, 
                                detect_anomaly=False,
                                deterministic=True,
                                )
        trainer.fit(self.module, dataloader)
        self.module.eval()

    def get_latent_original(self, adata=None):
        """generate and return the latent encodings for
            the original valued-spots

        Args:
            use_phase (str, optional): The latent of encodings to choose. 
                use "strc" to extract topology-based latent; "attr" to 
                extract expression-based latent; "comb" for the concated
                latent encodings ([z_strc, z_attr]). Defaults to "strc".

        Raises:
            ValueError: Wrong identifier of the 'use_phase'.

        Returns:
            z: latent embeddings for the original valued-spots.
        """
        z_attr, z_strc, _, _, _, _ = self._get_module_output()
        value_idx = self.gdata.value_idx
        x_id = self.gdata.x_id
        z_attr = z_attr.detach().numpy()
        z_strc = z_strc.detach().numpy()
        
        z_strc_ori = z_strc[value_idx]
        
        df_attr = pd.DataFrame(z_attr, index=x_id[value_idx])
        df_topo = pd.DataFrame(z_strc_ori, index=x_id[value_idx])
        
        if hasattr(self.gdata, "mask_idx"):
            mask_idx = self.gdata.mask_idx.numpy()
            mask_id = x_id[mask_idx]

            mask_attr = self.feature_propagate(node_embed=z_attr)[mask_idx]
            df_attr = pd.concat([df_attr, pd.DataFrame(mask_attr, index=mask_id)], axis=0)
            df_topo = pd.concat([df_topo, pd.DataFrame(z_strc[mask_idx], index=mask_id)], axis=0)
            
        df_comb = pd.concat([df_attr, df_topo], axis=1)
        
        if adata is not None:
            adata.obsm["STForte_ATTR"] = df_attr.loc[adata.obs_names, :].to_numpy()
            adata.obsm["STForte_TOPO"] = df_topo.loc[adata.obs_names, :].to_numpy()
            adata.obsm["STForte_COMB"] = df_comb.loc[adata.obs_names, :].to_numpy()
            if hasattr(self.gdata, "mask_idx"):
                adata.obs["STForte_Mask"] = pd.Categorical(list(map(lambda x: "Masked" if x in mask_id else "Unmasked", 
                                                                     adata.obs_names)))
        else:
            results = {
                "STForte_ATTR": df_attr.loc[adata.obs_names, :].to_numpy(),
                "STForte_TOPO": df_topo.loc[adata.obs_names, :].to_numpy(),
                "STForte_COMB": df_comb.loc[adata.obs_names, :].to_numpy(),
                "STForte_Mask": pd.Categorical(list(map(lambda x: "Masked" if x in mask_id else "Unmasked", 
                                                                     adata.obs_names))),
            }
            return results

    def get_result_anndata(self,
                             adj_mat: bool = True,
                             use_fp=True, fp_kwargs: dict = {}):
        """Retrieve the anndata including the inferred spots,
            coordinates and latent encodings.
           Do NOT use this method in uninferred situation.
        Args:
            with_expand (bool, optional): whether to inlcude spots
            generated by the expand topology . Defaults to True.

        Returns:
            (AnnData): {
                x: expression matrix (if with_expand = True, the new
                    spots are inferred with NB sampling).
                obs: spot identifiers (with preserving obs_names for
                        the original spots).
                var: gene identifiers that preserving original var_names.
                obsm: {'Z_TOPO': topology encodings, 'Z_ATTR_FP', attribute encodings.
                        encodings, 'coord': spatial coordinates}.
            }
        """

        x, value_idx, infer_idx = self.gdata.x, self.gdata.value_idx, self.gdata.infer_idx
        z_attr, z_strc, _, r_sa, _, _ = self._get_module_output()
        z_attr = z_attr.cpu().detach().numpy()
        z_strc = z_strc.cpu().detach().numpy()

        # get reconstruction output
        x_recon = r_sa.detach()
        var_identifier = pd.Series(["PC_{:d}".format(d + 1) for d in range(0, x_recon.shape[1])])
        x_recon = x_recon.numpy()
        # use feature propagation
        if use_fp:
            z_attr_fp = self.feature_propagate(z_attr, **fp_kwargs)
        # allocate valued/inferred attributes
        x = np.concatenate([x[value_idx].numpy(), x_recon[infer_idx]], axis=0)
        z_strc = np.concatenate([z_strc[value_idx], z_strc[infer_idx]], axis=0)
        mat_info = {'SP_TOPO':z_strc, 
                    'spatial':self.gdata.coord.cpu().numpy()}
        if use_fp:
            z_attr_fp = np.concatenate([z_attr_fp[value_idx], z_attr_fp[infer_idx]], axis=0)
            mat_info['SP_ATTR_FP'] = z_attr_fp
            mat_info['SP_COMB'] = np.concatenate([z_attr_fp, z_strc], axis=1)
        # construct anndata
        spot_id = np.concatenate([self.gdata.x_id[value_idx], self.gdata.x_id[infer_idx]])
        spot_df = pd.DataFrame(index=spot_id)
        spot_df['spot_instance'] = pd.Series(['Observed'] * len(value_idx) + ['Inferred'] * len(infer_idx), index=spot_id)
        
        adata_out = anndata.AnnData(
            X=x,
            obs=spot_df,
            var=pd.DataFrame(index=var_identifier),
            obsm=mat_info
        )
        if adj_mat:
            adata_out.obsp["A_STForte_recon"] = self.get_adjacent_matrix("strc", origin_only=False)
            adata_out.obsp["A_STForte_conn"] = self.gdata.adj_t.to_scipy(layout="csr")
        return adata_out

    def _get_module_output(self, gdata=None):
        self.module.eval()
        if gdata is None:
            gdata = self.gdata
        # x, edge_index, I = self.module.get_graph_essential(gdata)
        # value_idx = gdata.value_idx
        return self.module._process(gdata)

    def load_from_state_dict(self, path:str):
        self.module.load_state_dict(torch.load(path))

    def save_state_dict(
        self, 
        path: str, 
        alias: str = None,
        save_gdata: bool = True
        ):
        ''' 
            create a folder for model saving 
                --Path (user defiened):
                 |-- STForte_model_{time} # -> new folder created for model saving 
                    |-- model.pt  # model parameters (without model structure)
                    |-- hyper_params.pkl  # hyper-parameters used to reconstruct the model structure 
                    |-- gdata.pkl # gdata of the model

            other parameters:
                save_gdata: [bool] whether to save gdata of the model
                    ** Caution! STForte model initialized from saved model need gdata as necessary input! **
        '''
        # --- create folder for model saving --- #
        if alias is not None:
            rslt_path = os.path.join(path, alias)
        else: 
            time_ = time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())
            rslt_path = os.path.join(path, f'STForte_model_{time_}/')
        if not os.path.exists(rslt_path):
            os.makedirs(rslt_path)

        # --- save model parameters --- #
        
        torch.save(self.module.state_dict(), os.path.join(rslt_path, 'model.pt'))

        # --- save model hyper-parameters --- #
        with open(os.path.join(rslt_path, 'hyper_params.pkl'), 'wb') as f:
            pickle.dump(self.hyper_params, f)

        # --- save gdata --- #
        if save_gdata:
            with open(os.path.join(rslt_path, 'gdata.pkl'), "wb") as f:
                pickle.dump(self.gdata, f)

    def __slice_stdata__(cls, stdata:sc.AnnData, slice):
        if slice[0] == 'obs': tar = stdata.obs
        elif slice[0] == 'obsm': tar = stdata.obsm
        elif slice[0] == 'uns': tar = stdata.uns
        elif slice[0] == 'var': tar = stdata.var
        else: raise ValueError('Select slice of anndata from ["obs","var","obsm","uns"] !')

        if len(slice) == 2:
            return np.array(tar[slice[1]])
        elif len(slice) == 3:
            if isinstance(tar,pd.DataFrame):
                return np.array(tar[[slice[1],slice[2]]])
            elif isinstance(tar,Dict):
                return np.concatenate([np.array(tar[slice[1]]),np.array(tar[slice[2]])],axis=1)

    def _cluster_initialize(self, data, 
                            n_clusters: int = None, 
                            mb_kmeans: bool = False,
                            cluster_kwargs: dict = {},
                            use_mclust: bool = True,):
        z_attr, z_strc, _, _, _, _ = self.module._process(data)
        idx = data.value_idx
        z = self.module._extract_latent(z_attr, z_strc, idx, self.cluster_focus_phase)
        z = z.cpu().detach().numpy()
        if not use_mclust:
            if mb_kmeans:
                clustering = MiniBatchKMeans(n_clusters, random_state=42, **cluster_kwargs)
            else: 
                clustering = KMeans(n_clusters, random_state=42, **cluster_kwargs)
            y_pred_update = np.copy(clustering.fit_predict(z))
            c = clustering.cluster_centers_
        else:
            print("Using Mclust to initialize DEC params...")
            # Using MclustR to initiate DEC params
            np.random.seed(42)
            import rpy2.robjects as robjects
            robjects.r.library("mclust")

            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            r_random_seed = robjects.r['set.seed']
            r_random_seed(42)
            rmclust = robjects.r['Mclust']

            res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(z), n_clusters, 'EEE')
            y_pred_update = res[-2] # predicted labels
            c = res[-4][1].T
            print("DEC clustering initialization done.")
        
        return c, y_pred_update
    
    def feature_propagate(self, 
                          node_embed: np.ndarray,
                          device: str = "cuda", 
                          max_iter: int = 300):
        node_embed = torch.FloatTensor(node_embed)
        edge_idx = torch.stack(self.gdata.adj_t.coo()[:2], dim=0)
        inv_dist = self.gdata.inv_dist
        value_idx = self.gdata.value_idx
        n_nodes = self.gdata.x.shape[0]
        
        node_fp = FeaturePropagate(node_embed, edge_idx, inv_dist, value_idx, n_nodes,
                                   device=device, max_iter=max_iter)
        
        return node_fp.cpu().detach().numpy()

    def get_adjacent_matrix(self, use_phase: str = "strc", origin_only: bool = True, adata: AnnData = None):
        """Generate adjacent matrix by the STForte decoder

        Args:
            use_phase (str, optional): Phase of the output adjmatrix. "attr" for the attribute decoder output; "strc" for topology decoder output. 
            "neighbour to extract the input spatial topology. Defaults to "strc".
            origin_only (bool, optional): If use_phase="strc", It selects whether to preserve the original points only. Defaults to True.
            adata (AnnData, optional): If adata is assigned, the adjmatrix will be joined in adata.obsp, otherwise the matrix will be returned.
            Defaults to None.

        Raises:
            ValueError: Wrong identifier of the 'use_phase'.

        Returns:
            A (ndarray): If adata is not assigned, the adjmatrix will be returned as ndarray format.
        """
        _, _, _, _, A_attr, A_strc = self.module._process(self.gdata)
        A_attr = A_attr.detach().numpy()
        A_strc = A_strc.detach().numpy()
        v_idx = self.gdata.value_idx.numpy()
        if use_phase == "attr":
            if adata is not None:
                adata.obsp["STForte_CONN_ATTR"] = A_attr
            else:
                return A_attr
        elif use_phase == "strc":
            if origin_only:
                if adata is not None:
                    adata.obsp["STForte_CONN_STRC"] = A_strc[v_idx, :][:, v_idx]
                else:
                    return A_strc[v_idx, :][:, v_idx]
            else:
                if adata is not None:
                    adata.obsp["STForte_CONN_STRC"] = A_strc
                else:
                    return A_strc
        elif use_phase == "neighbour":
            if self.gdata is not None:
                return to_dense_adj(self.gdata.edge_index, edge_attr=self.gdata.edge_attr, max_num_nodes=self.gdata.x.shape[0]).squeeze()
            else:
                raise ValueError("model.gdata should be given when `use_phase == neighbour`.")
        else:
            raise ValueError("`use_phase` should be one of the identifiers from [`strc`, `attr`, `neighbour`].")
