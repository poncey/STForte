import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from typing import OrderedDict, Union


def lin_seq(input_dim: int, d_hidden: Union[int, list],
            d_output: int = None, act_out: type = None,
            act_hidden: type = nn.ReLU,
            dropout_rate: float = 0,
            act_hid_kwargs: dict = {},
            norm: type = None,
            alias: str = ''):
    """Generate layer sequence of linear combination.

    Args:
        input_dim (int): Dimension of the layer-sequence input;
        d_hidden (Union[int, list]): Dimension of idden layer(s). 
            the hidden layers will be script as the ordered num of 
            units if it's a list;
        d_output (int, optional): Dimension of layer output. 
            If not given, the final hidden layer stands for the output;
        act_out (type, optional): Activation for the output. By 
            default, nothing will be non-lineared for output;
        act_hidden (type, optional): Activation for the hidden. By
            default, it gives nn.ReLU;
        dropout_rate (float, optional): dropout rate of hidden layers. 
            Defaults to 0, which disable the dropout;
        act_hid_kwargs (dict, optional): arguments for hidden activation. 
            Defaults to {} with no args;
        norm (type, optional): Noramlization layer for the hidden (e.g., 
            nn.LayerNorm). By defaults, no normalization is used. 
        alias (str, optional): identifier of the linear component.
    Returns:
        nn.Module: Sequential linear components.
    """
    nn_units = [d_hidden] if isinstance(
        d_hidden, int) else d_hidden.copy()
    nn_units.insert(0, input_dim)
    lincomp = OrderedDict()
    alias = alias + '_' if alias != '' else alias
    for ii in range(len(nn_units) - 1):
        lincomp['{:s}dense_{:d}'.format(alias,
            ii + 1)] = nn.Linear(nn_units[ii], nn_units[ii + 1])
        lincomp['{:s}activ_{:d}'.format(alias,
            ii + 1)] = act_hidden(**act_hid_kwargs)
        if dropout_rate != 0:
            lincomp['{:s}dropout_{:d}'.format(alias,
                ii + 1)] = nn.Dropout(p=dropout_rate)
        if norm is not None:
            lincomp['{:s}batchnm_{:d}'.format(alias,
                ii + 1)] = norm(nn_units[ii + 1])
    if d_output is not None:
        lincomp['{:s}out'.format(alias)] = nn.Linear(nn_units[-1], d_output)
        if act_out is not None:
            lincomp['{:s}_act_out'.format(alias)] = act_out()
    return nn.Sequential(lincomp)


def gcn_seq(input_dim: int, d_hidden: Union[int, list],
            d_output: int = None, act_out: type = None,
            act_hidden: type = nn.ReLU, act_hid_kwargs: dict = {},
            norm: type = None, dropout_rate: float = 0,
            base_unit: MessagePassing = gnn.GCNConv,
            gnn_kwargs: dict = {}):
    """Generate layer sequence of GCN combination.

    Args:
        input_dim (int): Dimension of the layer-sequence input;
        d_hidden (Union[int, list]): Dimension of idden layer(s). 
            the hidden layers will be script as the ordered num of 
            units if it's a list;
        d_output (int, optional): Dimension of layer output. 
            If not given, the final hidden layer stands for the output;
        act_out (type, optional): Activation for the output. By 
            default, nothing will be non-lineared for output;
        act_hidden (type, optional): Activation for the hidden. By
            default, it gives nn.ReLU;
        dropout_rate (float, optional): dropout rate of hidden layers. 
            Defaults to 0, which disable the dropout;
        act_hid_kwargs (dict, optional): arguments for hidden activation. 
            Defaults to {} with no args;
        norm (type, optional): Noramlization layer for the hidden (e.g., 
            nn.LayerNorm). By defaults, no normalization is used. 
        gnn_kwargs (dict, optional): arguments of graph neural network.
    Returns:
        nn.Module: Sequential linear components.
    """
    gnn_units = [d_hidden] if isinstance(d_hidden, int) else d_hidden.copy()
    gnn_units.insert(0, input_dim)
    gcncomp = []
    for ii in range(len(gnn_units) - 1):
        gcncomp.append(
            (base_unit(gnn_units[ii], gnn_units[ii + 1], **gnn_kwargs), 
             'x, edge_index -> x')
        )
        gcncomp.append(act_hidden(**act_hid_kwargs))
        if dropout_rate != 0:
            gcncomp.append(nn.Dropout(p=dropout_rate))
        if norm is not None:
            gcncomp.append(norm(gnn_units[ii + 1]))
    if d_output is not None:
        gcncomp.append(
            (base_unit(gnn_units[-1], d_output, **gnn_kwargs),
             'x, edge_index -> x')
        )
    if act_out is not None:
        gcncomp.append(act_out())
    return gnn.Sequential('x, edge_index', gcncomp)


def gat_seq(input_dim: int, d_hidden: Union[int, list],
            d_output: int = None, act_out: type = None,
            act_hidden: type = nn.ReLU, act_hid_kwargs: dict = {},
            norm: type = None, dropout_rate: float = 0,
            base_unit: MessagePassing = gnn.GATConv, edge_dim: int = 1,
            gnn_kwargs: dict = {}):
    """Generate layer sequence of GAT/GraphTransformer combination.

    Args:
        input_dim (int): Dimension of the layer-sequence input;
        d_hidden (Union[int, list]): Dimension of idden layer(s). 
            the hidden layers will be script as the ordered num of 
            units if it's a list;
        d_output (int, optional): Dimension of layer output. 
            If not given, the final hidden layer stands for the output;
        act_out (type, optional): Activation for the output. By 
            default, nothing will be non-lineared for output;
        act_hidden (type, optional): Activation for the hidden. By
            default, it gives nn.ReLU;
        dropout_rate (float, optional): dropout rate of hidden layers. 
            Defaults to 0, which disable the dropout;
        act_hid_kwargs (dict, optional): arguments for hidden activation. 
            Defaults to {} with no args;
        norm (type, optional): Noramlization layer for the hidden (e.g., 
            nn.LayerNorm). By defaults, no normalization is used. 
        gnn_kwargs (dict, optional): arguments of graph neural network.
    Returns:
        nn.Module: Sequential linear components.
    """
    gnn_units = [d_hidden] if isinstance(d_hidden, int) else d_hidden.copy()
    gnn_units.insert(0, input_dim)
    gatcomp = []
    for ii in range(len(gnn_units) - 1):
        gatcomp.append(
            (base_unit(gnn_units[ii], gnn_units[ii + 1], edge_dim=edge_dim, **gnn_kwargs), 
            'x, edge_index, edge_attr -> x')
        )
        gatcomp.append(act_hidden(**act_hid_kwargs))
        if dropout_rate != 0:
            gatcomp.append(nn.Dropout(p=dropout_rate))
        if norm is not None:
            gatcomp.append(norm(gnn_units[ii + 1]))
    if d_output is not None:
        gatcomp.append(
            (base_unit(gnn_units[-1], d_output, edge_dim=edge_dim, **gnn_kwargs),
             'x, edge_index, edge_attr -> x')
        )
    if act_out is not None:
        gatcomp.append(act_out())
    return gnn.Sequential('x, edge_index, edge_attr', gatcomp)


def FeaturePropagate(
    embeddings: torch.FloatTensor,
    edge_list: torch.LongTensor,
    edge_weight: torch.FloatTensor,
    pos_idx: torch.LongTensor,
    n_nodes: int,
    max_iter: int = 300,
    device: str = "cuda:0"
    )->torch.FloatTensor:
    edge_list = edge_list.T
    edge_weight = edge_weight.squeeze()
    pos_idx = pos_idx.to(device)
    embeddings = embeddings.to(device)

    # get the adjacency matrix and degree matrix
    D_negsqrt = scatter_add(edge_weight,edge_list[:,0],dim=0,dim_size=n_nodes).pow(-1/2)
    D_negsqrt[D_negsqrt==torch.inf] = 0
    DAD = D_negsqrt[edge_list[:,0]]*edge_weight*D_negsqrt[edge_list[:,1]]
    norm_adj = torch.sparse_coo_tensor(
        indices=edge_list.T, values=DAD, size=(n_nodes,n_nodes),dtype=torch.float32
    ).to(device)

    # initialize result
    result = torch.zeros((n_nodes,embeddings.size(1))).to(device)
    result[pos_idx] = embeddings

    # propagation
    for _ in range(max_iter):
        result = torch.sparse.mm(norm_adj,result)
        result[pos_idx] = embeddings
    return result
