import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class ThreeStageGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Stage 1: Pin -> Net
        self.conv_pin2net = dglnn.GraphConv(hidden_dim, hidden_dim, norm='right', allow_zero_in_degree=True)
        # Stage 2: Net <-> Net
        self.conv_net2net = dglnn.GraphConv(hidden_dim, hidden_dim, norm='right', weight=True,
                                            allow_zero_in_degree=True)
        # Stage 3: Net -> Pin
        self.conv_net2pin = dglnn.GraphConv(hidden_dim, hidden_dim, norm='right', allow_zero_in_degree=True)

        self.norm_net_1 = nn.LayerNorm(hidden_dim)
        self.norm_net_2 = nn.LayerNorm(hidden_dim)
        self.norm_pin = nn.LayerNorm(hidden_dim)

    def forward(self, g, h_pin, h_net, overlap_weights):

        # 1. Pin -> Net
        subg_p2n = g[('pin', 'connected', 'net')]
        res_net_1 = self.conv_pin2net(subg_p2n, (h_pin, h_net))
        h_net = self.norm_net_1(h_net + res_net_1)
        h_net = F.relu(h_net)

        # 2. Net <-> Net
        subg_n2n = g[('net', 'overlap', 'net')]
        res_net_2 = self.conv_net2net(subg_n2n, h_net, edge_weight=overlap_weights)
        h_net = self.norm_net_2(h_net + res_net_2)
        h_net = F.relu(h_net)

        # 3. Net -> Pin
        subg_n2p = g[('net', 'connected', 'pin')]
        res_pin = self.conv_net2pin(subg_n2p, (h_net, h_pin))
        h_pin = self.norm_pin(h_pin + res_pin)
        h_pin = F.relu(h_pin)

        return h_pin, h_net


class NetPredictor(nn.Module):
    def __init__(self, pin_in_dim, hidden_dim, out_dim, n_layers=3, dropout=0.2):
        super().__init__()
        self.pin_projector = nn.Linear(pin_in_dim, hidden_dim)
        self.layers = nn.ModuleList([ThreeStageGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, g, pin_feats, overlap_weights=None):
        h_pin = F.relu(self.pin_projector(pin_feats))

        # 在 Batch 模式下，num_nodes('net') 返回的是整个 Batch 中所有 Net 的总数
        h_net = torch.zeros((g.num_nodes('net'), h_pin.shape[1]), device=h_pin.device)

        for layer in self.layers:
            h_pin, h_net = layer(g, h_pin, h_net, overlap_weights)

        prediction = self.predictor(h_net)
        return prediction, h_net
