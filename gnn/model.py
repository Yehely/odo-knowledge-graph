"""
OpioidGNN — bipartite heterogeneous GNN for compound-target pChEMBL regression.

Architecture:
  1. Compound encoder:  Linear(F_c, H) -> BN -> ReLU -> Dropout
  2. Target encoder:    Linear(F_t_cat + emb_dim, H) -> BN -> ReLU -> Dropout
  3. Bipartite message passing: HeteroConv(SAGEConv) x n_layers
     compound->target  and  target->compound (reverse edges)
  4. Edge prediction head: MLP([h_c || h_t || edge_attr] -> 256 -> 128 -> 1)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import COMPOUND_HIDDEN, DROPOUT, GNN_LAYERS


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dims: list, dropout: float = DROPOUT):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OpioidGNN(nn.Module):
    """
    Parameters
    ----------
    compound_dim   : input feature dimension for compound nodes
    target_cat_dim : one-hot categorical dimension for target nodes
    n_targets      : number of unique targets (for learned embedding)
    edge_dim       : edge feature dimension
    hidden         : hidden dimension
    n_layers       : number of GNN message-passing layers
    emb_dim        : dimension of target name embedding
    dropout        : dropout probability
    """

    def __init__(
        self,
        compound_dim: int,
        target_cat_dim: int,
        n_targets: int,
        edge_dim: int,
        hidden: int = COMPOUND_HIDDEN,
        n_layers: int = GNN_LAYERS,
        emb_dim: int = 32,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.compound_encoder = nn.Sequential(
            nn.Linear(compound_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.target_embedding = nn.Embedding(n_targets, emb_dim)
        self.target_encoder = nn.Sequential(
            nn.Linear(target_cat_dim + emb_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv(
                {
                    ("compound", "activity", "target"): SAGEConv(
                        (hidden, hidden), hidden
                    ),
                    ("target", "rev_activity", "compound"): SAGEConv(
                        (hidden, hidden), hidden
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.bns_compound = nn.ModuleList(
            [nn.BatchNorm1d(hidden) for _ in range(n_layers)]
        )
        self.bns_target = nn.ModuleList(
            [nn.BatchNorm1d(hidden) for _ in range(n_layers)]
        )

        head_in = hidden + hidden + edge_dim
        self.edge_head = MLP([head_in, 256, 128, 1], dropout=dropout)

    # -----------------------------------------------------------------------

    def encode_nodes(self, data):
        h_c = self.compound_encoder(data["compound"].x)
        tgt_emb = self.target_embedding(data["target"].node_idx)
        h_t = self.target_encoder(
            torch.cat([data["target"].x, tgt_emb], dim=-1)
        )
        return h_c, h_t

    def message_pass(self, h_c, h_t, data):
        x_dict = {"compound": h_c, "target": h_t}
        edge_index_dict = {
            ("compound", "activity", "target"): (
                data[("compound", "activity", "target")].edge_index
            ),
            ("target", "rev_activity", "compound"): (
                data[("target", "rev_activity", "compound")].edge_index
            ),
        }

        for i, conv in enumerate(self.convs):
            x_new = conv(x_dict, edge_index_dict)
            x_dict["compound"] = F.dropout(
                F.relu(self.bns_compound[i](x_new["compound"])),
                p=self.dropout,
                training=self.training,
            )
            x_dict["target"] = F.dropout(
                F.relu(self.bns_target[i](x_new["target"])),
                p=self.dropout,
                training=self.training,
            )

        return x_dict["compound"], x_dict["target"]

    def predict_edges(self, h_c, h_t, edge_index, edge_attr):
        src = h_c[edge_index[0]]
        dst = h_t[edge_index[1]]
        e = torch.cat([src, dst, edge_attr], dim=-1)
        return self.edge_head(e).squeeze(-1)

    def forward(self, data, edge_index=None, edge_attr=None):
        h_c, h_t = self.encode_nodes(data)
        h_c, h_t = self.message_pass(h_c, h_t, data)

        if edge_index is None:
            et = ("compound", "activity", "target")
            edge_index = data[et].edge_index
            edge_attr = data[et].edge_attr

        return self.predict_edges(h_c, h_t, edge_index, edge_attr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(data, edge_dim: int) -> OpioidGNN:
    compound_dim = data["compound"].x.shape[1]
    target_cat_dim = data["target"].x.shape[1]
    n_targets = data["target"].num_nodes

    model = OpioidGNN(
        compound_dim=compound_dim,
        target_cat_dim=target_cat_dim,
        n_targets=n_targets,
        edge_dim=edge_dim,
    )

    # Shift output bias to the training-set mean pChEMBL so the model starts
    # near the correct value range from epoch 1. Weights stay at default
    # Kaiming init so gradients flow normally through all layers.
    et = ("compound", "activity", "target")
    train_mean = float(
        data[et].edge_label[data[et].train_mask].mean()
    )
    model.edge_head.net[-1].bias.data.fill_(train_mean)

    return model
