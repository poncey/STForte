import torch
import warnings
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.utils.data
import pytorch_lightning as pl
from torch.optim import Adam
from sklearn.utils import shuffle
from typing import Dict, OrderedDict, Union
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch_geometric.nn.conv import MessagePassing, PDNConv
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
    subgraph,
    to_dense_adj
)
from .__base import lin_seq, gat_seq, gcn_seq

class STForteModule(pl.LightningModule):
    def __init__(self,
                 d_attrs: int,
                 lr: float = 0.001,
                 pre_train_epochs: int = 0,
                 hidden_ftrs: Union[list, int] = [200, 50],
                 hidden_gcns: Union[list, int] = [200, 50],
                 d_latent: int = 32,
                 p_dropout: float = 0.1,
                 activation_kwargs: Dict = {},
                 # get partial adjacent for large data
                 partial_adjacent: bool = False,
                 # optimization components
                 lmbd_recon: float = 1.0,
                 lmbd_cross: float = 10.0,
                 lmbd_gan: float = 4.0,
                 steps_gen: int = 1,
                 steps_disc: int = 1,
                 # messsage passing settings
                 use_edge_attr: bool = False,
                 base_unit: Union[str, MessagePassing] = "PDN",
                 gnn_kwargs: Dict = dict(hidden_channels=1, edge_dim=1),
                 ):
        super().__init__()

        self.save_hyperparameters()

        # model parameters
        self.d_attrs = d_attrs
        self.d_latent = d_latent
        self.lr = lr
        self.pretrain_epochs = pre_train_epochs
        Activation = nn.ReLU

        # loss parameters
        self.lmbd_recon = lmbd_recon
        self.lmbd_cross = lmbd_cross
        self.lmbd_gan = lmbd_gan

        # training parameters

        if lmbd_gan > 0:
            self.steps_gen = steps_gen
        else:
            self.steps_gen = 1
        self.steps_disc = steps_disc
        self.partial_adjacent = partial_adjacent
        self.prior = Normal(loc=torch.FloatTensor([0.0]), scale=torch.FloatTensor([1.0]))

        # training configuration
        self.automatic_optimization = False  # enable manual optimization

        if hidden_gcns is None:
            hidden_gcns = hidden_ftrs
            
        # encoder for ae
        self.attr_encoder = lin_seq(self.d_attrs, hidden_ftrs, d_latent,
                                    dropout_rate=p_dropout,
                                    act_hidden=Activation, act_hid_kwargs=activation_kwargs,
                                    alias='attrenc')

        # encoder for gae
        gcn_in_dim = self.d_attrs
        self.use_edge_attr = use_edge_attr
        if isinstance(base_unit, str):
            if base_unit == "GCN":
                base_unit = gnn.GCNConv
            elif base_unit == "GAT":
                base_unit = gnn.GATConv
            elif base_unit == "PDN":
                base_unit = PDNConv
            else:
                raise ValueError("The `base_unit` should be `GCN` or `GAT`.")
        else:
            warnings.warn("This warning is to inform that the `base_unit` should be either string or the component from `torch_geometric.nn`")
        if not use_edge_attr:
            self.strc_encoder = gcn_seq(gcn_in_dim, hidden_gcns, d_latent,
                                        dropout_rate=p_dropout,
                                        act_hidden=Activation, act_hid_kwargs=activation_kwargs,
                                        base_unit=base_unit,
                                        gnn_kwargs=gnn_kwargs)
        else:
            self.strc_encoder = gat_seq(gcn_in_dim, hidden_gcns, d_latent,
                                        dropout_rate=p_dropout,
                                        act_hidden=Activation, act_hid_kwargs=activation_kwargs,
                                        base_unit=base_unit, edge_dim=1,
                                        gnn_kwargs=gnn_kwargs)

        # decoder for attributes
        self.attr_decoder = lin_seq(d_latent, hidden_ftrs[::-1], self.d_attrs,
                                    dropout_rate=p_dropout,
                                    act_hidden=Activation, act_hid_kwargs=activation_kwargs,
                                    alias='attrdec')

        # decoder for structure
        self.strc_decoder = nn.Sequential(
            OrderedDict([
                ('strcdec_dense1', nn.Linear(d_latent, d_latent)),
                ('strcdec_act1', Activation(**activation_kwargs)),
                ('strcdec_dp1', nn.Dropout(p=p_dropout)),
                ('strcdec_dense2', nn.Linear(d_latent, d_latent))]))

        # discriminator
        self.discriminator = nn.Sequential(
            OrderedDict([
                ('disc_dense1', nn.Linear(d_latent, d_latent)),
                ('disc_act1', Activation(**activation_kwargs)),
                ('disc_dp1', nn.Dropout(p_dropout)),
                ('disc_dense2', nn.Linear(d_latent, 1)),
                ('disc_sigmoid', nn.Sigmoid())
            ])
        )

    def forward(self, x, adj_t, I):
        # encode
        z_attr = self.attr_encoder(x)
        z_strc = self.strc_encoder(I, adj_t)
        # decode
        r_aa = self._decode_attr(z_attr)
        r_sa = self._decode_attr(z_strc)
        r_as = self._decode_strc(z_attr, sigmoid=True)
        r_ss = self._decode_strc(z_strc, sigmoid=True)

        return z_attr, z_strc, r_aa, r_sa, r_as, r_ss
    
    def forward_partial(self, x, adj_t, I):
        # encode
        z_attr = self.attr_encoder(x)
        z_strc = self.strc_encoder(I, adj_t)
        # decode
        r_aa = self._decode_attr(z_attr)
        r_sa = self._decode_attr(z_strc)
        z_as = self._decode_strc(z_attr, sigmoid=True, get_latent_only=True)
        z_ss = self._decode_strc(z_strc, sigmoid=True, get_latent_only=True)
        return z_attr, z_strc, r_aa, r_sa, z_as, z_ss

    def _extract_latent(self, z_attr, z_strc, idx, phase):
        if phase == "strc":
            return z_strc[idx]
        elif phase == "strc_all":
            return z_strc
        elif phase == "attr":
            return z_attr
        else:
            return torch.cat([z_strc[idx], z_attr], dim=1)

    def _process(self, data):
        # TODO: for partial return
        x, adj_t, node_feature = self.get_graph_essential(data)
        idx = data.value_idx
        x_attr = x[idx]
        if self.partial_adjacent:
            return self.forward_partial(x_attr, adj_t, node_feature)
        else:
            return self.forward(x_attr, adj_t, node_feature)   

    def training_step(self, batch):
        opt_gen, opt_disc = self.optimizers()
        data = batch.to(self.device)
        # train generators
        for _ in range(1, self.steps_gen + 1):
            opt_gen.zero_grad()
            joint_loss, G_loss = self._step_gen(data)
            gen_loss = joint_loss
            self.log("joint_loss_train", joint_loss.item())
            if self.current_epoch > self.pretrain_epochs:
                gen_loss += G_loss
                self.log("G_loss_train", 0 if self.lmbd_gan==0 else G_loss.item() / self.lmbd_gan)
            gen_loss.backward()
            opt_gen.step()
            self.log("gen_loss_train", gen_loss.item())

        # train discriminator
        if self.lmbd_gan > 0:
            if self.current_epoch > self.pretrain_epochs:
                for _ in range(1, self.steps_disc + 1):
                    opt_disc.zero_grad()
                    D_loss = self._step_disc(data)
                    D_loss.backward()
                    opt_disc.step()
                    self.log("D_loss_train", 0 if self.lmbd_gan==0 else D_loss.item() / self.lmbd_gan)

    def configure_optimizers(self):
        params_gen = list(self.attr_encoder.parameters()) +\
                       list(self.strc_encoder.parameters()) +\
                       list(self.attr_decoder.parameters()) +\
                       list(self.strc_decoder.parameters())                           
        opt_gen = Adam(params_gen, lr=self.lr)
        opt_disc = Adam(self.discriminator.parameters(), lr=self.lr)
        return opt_gen, opt_disc

    def _decode_strc(self, z, sigmoid: bool = True, get_latent_only: bool = False):
        z = self.strc_decoder(z)
        if get_latent_only:
            return z
        else:
            if sigmoid:
                return torch.sigmoid(torch.mm(z, z.t()))
            else:
                return torch.mm(z, z.t())

    def _decode_attr(self, z):
        return self.attr_decoder(z)

    def _step_gen(self, data, eps=1e-16):

        # SAT loss
        x, adj_t, node_feature = self.get_graph_essential(data)
        idx = data.value_idx

        x_attr = x[idx]

        if self.partial_adjacent:
            z_attr, z_strc, r_aa, r_sa, z_as, z_ss = self.forward_partial(x_attr, adj_t, node_feature)
        else:
            z_attr, z_strc, r_aa, r_sa, r_as, r_ss = self.forward(x_attr, adj_t, node_feature)
        pos_edge_idx = torch.stack(adj_t.coo()[:2], dim=0)
        sub_pos_edge_idx, _ = subgraph(idx, pos_edge_idx, data.edge_attr)
        # reconstruction loss
        loss_aa = self.lmbd_recon * self._attr_recon_loss(r_aa, x[idx])
        loss_sa = self.lmbd_cross * self._attr_recon_loss(r_sa[idx], x[idx])
        if self.partial_adjacent:
            loss_as = self.lmbd_cross * self._strc_recon_loss_partial(z_as, pos_edge_idx=sub_pos_edge_idx)
            loss_ss = self.lmbd_recon * self._strc_recon_loss_partial(z_ss, pos_edge_idx=sub_pos_edge_idx)
        else:
            neg_edge_idx = torch.nonzero(to_dense_adj(pos_edge_idx).squeeze()==0, as_tuple=False).T
            sub_neg_edge_idx, _ = subgraph(idx, neg_edge_idx)
            neg_edge_idx = None
            sub_neg_edge_idx = None
            loss_as = self.lmbd_cross * self._strc_recon_loss(r_as, pos_edge_idx=sub_pos_edge_idx, neg_edge_idx=sub_neg_edge_idx)
            loss_ss = self.lmbd_recon * self._strc_recon_loss(r_ss, pos_edge_idx=pos_edge_idx, neg_edge_idx=neg_edge_idx)

        # G loss
        if self.lmbd_gan > 0 and self.current_epoch > self.pretrain_epochs:
            G_loss = (
                -torch.log(self.discriminator(z_strc) + eps).mean() \
                -torch.log(self.discriminator(z_attr) + eps).mean()
                ) * self.lmbd_gan
        else:
            G_loss = torch.FloatTensor([0]).to(self.device)
        joint_loss = loss_aa + loss_sa + loss_as + loss_ss

        return joint_loss, G_loss

    def _step_disc(self, data, eps=1e-16):
        z_attr, z_strc, _, _, _, _ = self._process(data)
        idx = data.value_idx
        z_norm = self._sample_prior_like(z_attr).to(self.device)
        D_loss = -(torch.log(self.discriminator(z_norm) + eps).mean() * 2 +\
            torch.log(1 - self.discriminator(z_attr) + eps).mean() +\
            torch.log(1 - self.discriminator(z_strc[idx]) + eps).mean()
            ) * self.lmbd_gan
        return D_loss
 
    def _sample_prior_like(self, z):
        """sample a gaussion prior with same shape of the input

        Args:
            z : input Tensor

        Returns:
            z_g: sample of standard normal prior with same shape of z.
        """
        return self.prior.sample([z.shape[0], z.shape[1]]).reshape([z.shape[0], z.shape[1]])

    def _strc_recon_loss(self, r_g, pos_edge_idx=None, neg_edge_idx=None, eps=1e-16):
        """reconstruction loss of graph structure (adjcent matrix)

        Args:
            r_g : the generated adjecent matrix (sigmoid output, FloatTensor);
            pos_edge_idx : indices of the positive edges
            neg_edge_idx : indices of the negtaive edges. If not given, it
                           will try to infer the neg_edge_indx from data.
            eps : error tolerence of log-function. Defaults to 1e-16.

        Returns:
            strc_loss: BCE reconstruction loss of edges.
        """
        logit_pos = r_g[pos_edge_idx[0], pos_edge_idx[1]] + eps
        pos_loss = -(torch.log(logit_pos)).mean()
        # prevent self-loops within negative samples
        pos_edge_idx, _ = remove_self_loops(pos_edge_idx)
        pos_edge_idx, _ = add_self_loops(pos_edge_idx)
        if neg_edge_idx is None:
            neg_edge_idx = negative_sampling(pos_edge_idx, r_g.size(0))
        logit_neg = 1 - r_g[neg_edge_idx[0], neg_edge_idx[1]] + eps
        neg_loss = -(torch.log(logit_neg)).mean()
        
        return pos_loss + neg_loss

    def _strc_recon_loss_partial(self, z, pos_edge_idx=None, neg_edge_idx=None, eps=1e-16):
        r_pos = torch.sigmoid((z[pos_edge_idx[0]] * z[pos_edge_idx[1]]).sum(dim=1))
        pos_loss = -torch.log(r_pos + eps).mean()
        # prevent self-loops within negative samples
        pos_edge_idx, _ = remove_self_loops(pos_edge_idx)
        pos_edge_idx, _ = add_self_loops(pos_edge_idx)
        if neg_edge_idx is None:
            neg_edge_idx = negative_sampling(pos_edge_idx, z.size(0))
        r_neg = torch.sigmoid((z[neg_edge_idx[0]] * z[neg_edge_idx[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - r_neg + eps).mean()
        return pos_loss + neg_loss

    def _attr_recon_loss(self, r, x):
        """attribute reconsturction loss for gene expression

        Args:
            r : decoder output, (|V|, d)
            x : real expr. value (|V|, d)

        Returns:
            attr_loss: NB neg-likelihood of node attributes.
        """
        loss = F.mse_loss(x, r)
        return loss

    def get_graph_essential(self, data):
            return data.x, data.adj_t, data.xfp

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

