import numpy as np
import pandas as pd
from typing import Union
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from anndata._core.anndata import AnnData

            
def complete_unseen_expression(adata_p: AnnData,
                               gene_name: Union[str, list],
                               adata_o: AnnData,
                               latent_embed: Union[str, pd.Series, np.ndarray] = "SP_TOPO",
                               indexer_ori: Union[str, pd.Series, list] = "spot_instance",
                               predictor: str = "xgboost",
                               # xgboost parameters
                               xgb_params: dict = {"n_estimators":150},
                               truncate_negative: bool = True,
                               name_suffix="_with_padding",
                               layer=None,
                               copy: bool = False) -> Union[None, pd.DataFrame]:
    # Get latent topology embeddings 
    ind = _get_indexer(adata_p, indexer_ori)
    ind_val = adata_p.obs[ind == True].index
    ind_pad = adata_p.obs[ind == False].index
    z_val = _get_latent(adata_p, ind_val, latent_embed)
    z_pad = _get_latent(adata_p, ind_pad, latent_embed)
    if isinstance(gene_name, str):
        gene_name = [gene_name] 
    df_all = []
    # Prediction
    for gn in tqdm(gene_name, desc='Propagating genes...', leave=False):
        if layer is None:
            y_val = adata_o[ind_val][:, gn].X.todense().A.squeeze()
        else:
            try:
                y_val = adata_o[ind_val][:, gn].layers[layer].squeeze()
            except AttributeError:
                y_val = adata_o[ind_val][:, gn].layers[layer].A.squeeze()
        if predictor == "xgboost":
            reg = XGBRegressor(**xgb_params)
            reg.fit(z_val, y_val)
            y_pad = reg.predict(z_pad)
        else:
            raise ValueError("Wrong identifier for the predictor. It should be \"xgboost\" or \"deepforest\".")
        df_val = pd.DataFrame(y_val, index=ind_val, columns=["{:s}{:s}".format(gn, name_suffix)])
        df_pad = pd.DataFrame(y_pad, index=ind_pad, columns=["{:s}{:s}".format(gn, name_suffix)])
        if truncate_negative:
            df_pad[df_pad <= 0] = 0  # truncate negative values into zero.
        if copy:
            df_all.append(pd.concat([df_val, df_pad]))
        else:
            adata_p.obs["{:s}_with_padding".format(gn)] = pd.concat([df_val, df_pad])
    if copy:
        return pd.concat(df_all, axis=1)


def annotation_propagate(adata_p: AnnData,
                         annotation: str,
                         adata_o: AnnData,
                         latent_embed: Union[str, pd.Series, np.ndarray] = "SP_TOPO",
                         indexer_ori: Union[str, pd.Series, list] = "spot_instance",
                         predictor:str = "xgboost",
    
                        # xgboost parameters
                        xgb_params: dict = {"n_estimators":150},
                        
                        copy: bool = False) -> Union[None, pd.DataFrame]:
    
    # Get latent topology embeddings 
    # TODO: observed, unobserved related.
    
    
    ind = _get_indexer(adata_p, indexer_ori)
    ind_val = adata_p.obs[ind == True].index
    ind_pad = adata_p.obs[ind == False].index
    z_val = _get_latent(adata_p, ind_val, latent_embed)
    z_pad = _get_latent(adata_p, ind_pad, latent_embed)
    
    # Get annotation labels
    y_val = adata_o.obs[annotation][ind == True].to_numpy()
    # If label not start from (include) zero, using LabelEncoder
    le = LabelEncoder()
    y_val = le.fit_transform(y_val)
    
    # print(f'zval shape: {z_val.shape}, yval shape: {y_val.shape}')
    # Prediction
    if predictor == "xgboost":
        reg = XGBClassifier(**xgb_params)
        reg.fit(z_val, y_val)
        y_pad = reg.predict(z_pad)
    elif predictor == "gnn":
        pass
    elif predictor == "label_propagation":
        pass
    else:
        raise ValueError("Wrong identifier for the predictor. It should be \"xgboost\" or \"gnn\". or \"label_propagation\"")
    # label transform
    y_val = le.inverse_transform(y_val)
    y_pad = le.inverse_transform(y_pad)
    df_val = pd.DataFrame(y_val, index=ind_val, columns=["{:s}_with_padding".format(annotation)])
    df_pad = pd.DataFrame(y_pad, index=ind_pad, columns=["{:s}_with_padding".format(annotation)])
    if copy:
        return pd.concat([df_val, df_pad]).astype("category")
    else:
        adata_p.obs["{:s}_with_padding".format(annotation)] = pd.concat([df_val, df_pad]).astype("category")


def _get_indexer(adata, indexer, 
                 pos_identifier: str = 'Observed', neg_identifier: str = 'Inferred'):
    if isinstance(indexer, str):
        indexer = adata.obs[indexer]
    indexer = indexer.replace([pos_identifier, neg_identifier], [True, False])
    return indexer
    
    
def _get_latent(adata, ind, latent_embed):
    if isinstance(latent_embed, str):
        return adata[ind].obsm[latent_embed]
    else:
        return latent_embed[ind, :]


def complete_unseen_properties(adata_p: AnnData,
                               properties: Union[str, list],
                               adata_o: AnnData,
                               latent_embed: Union[str, pd.Series, np.ndarray] = "SP_TOPO",
                               indexer_ori: Union[str, pd.Series, list] = "spot_instance",
                               predictor: str = "xgboost",
                               # xgboost parameters
                               xgb_params: dict = {"n_estimators":150},
                               truncate_negative: bool = True,
                               copy: bool = False) -> Union[None, pd.DataFrame]:
    # Get latent topology embeddings 
    ind = _get_indexer(adata_p, indexer_ori)
    ind_val = adata_p.obs[ind == True].index
    ind_pad = adata_p.obs[ind == False].index
    z_val = _get_latent(adata_p, ind_val, latent_embed)
    z_pad = _get_latent(adata_p, ind_pad, latent_embed)
    if isinstance(properties, str):
        properties = [properties] 
    df_all = []
    # Prediction
    for pp in properties:
        y_val = adata_o[ind_val].obs[pp].squeeze()
        if predictor == "xgboost":
            reg = XGBRegressor(**xgb_params)
            reg.fit(z_val, y_val)
            y_pad = reg.predict(z_pad)
        else:
            raise ValueError("Wrong identifier for the predictor. It should be \"xgboost\" or \"deepforest\".")
        df_val = pd.DataFrame(y_val, index=ind_val, columns=["{:s}_with_padding".format(pp)])
        df_pad = pd.DataFrame(y_pad, index=ind_pad, columns=["{:s}_with_padding".format(pp)])
        if truncate_negative:
            df_pad[df_pad <= 0] = 0  # truncate negative values into zero.
        if copy:
            df_all.append(pd.concat([df_val, df_pad]))
        else:
            adata_p.obs["{:s}_with_padding".format(pp)] = pd.concat([df_val, df_pad])
    if copy:
        return pd.concat(df_all, axis=1)