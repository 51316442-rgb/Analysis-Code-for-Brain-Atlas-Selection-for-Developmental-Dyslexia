import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class AttentionFusion(nn.Module):
    def __init__(self, local_dim=6, global_dim=10, d=12):
        super().__init__()
        self.W_l = nn.Linear(local_dim, d)
        self.W_g = nn.Linear(global_dim, d)
        self.W_a = nn.Linear(d, 1)

    def forward(self, x_local, x_glob):
        B, N, _ = x_local.shape
        H_local = F.relu(self.W_l(x_local))                  # (B, N, d)
        h_glob = F.relu(self.W_g(x_glob))                    # (B, d)
        H_glob = h_glob.unsqueeze(1).repeat(1, N, 1)         # broadcast to N nodes
        alpha = torch.sigmoid(self.W_a(torch.tanh(H_local + H_glob)))
        return alpha * H_local + (1 - alpha) * H_glob        # (B, N, d)


class BrainGCNClassifier(nn.Module):
    def __init__(self, local_dim=6, global_dim=10, d_embed=12, hidden_dim=64,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.fusion = AttentionFusion(local_dim, global_dim, d_embed)
        self.gcn1 = GCNConv(d_embed, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, batch, edge_index = data.x, data.batch, data.edge_index
        global_feat = data.global_feat                      # (B, d_g)
        B = global_feat.size(0)
        N = x.size(0) // B
        x = x.view(B, N, -1)
        x = self.fusion(x, global_feat)                     # (B, N, d)
        x = x.view(-1, x.size(-1))
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)
