import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

GRAPH_THRESHOLD = 0.0


class BrainGraphDataset(Dataset):
    def __init__(self, local_feat, global_feat, adj_matrix, labels,
                 threshold=GRAPH_THRESHOLD, self_loops=False):
        """local_feat (N,90,6), global_feat (N,22), adj_matrix (N,90,90), labels (N,)."""
        super().__init__()
        self.local_feat = torch.tensor(local_feat, dtype=torch.float32)
        self.global_feat = torch.tensor(global_feat, dtype=torch.float32)
        self.adj_matrix = adj_matrix
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
        self.threshold = threshold
        self.self_loops = self_loops
        self._graphs = [self._build_graph(i) for i in range(len(self.labels))]

    def _build_graph(self, i):
        adj = np.asarray(self.adj_matrix[i])
        if self.threshold > 0:
            mask = adj > self.threshold
        else:
            mask = adj > 0.0
        if self.self_loops:
            np.fill_diagonal(mask, True)
        edge_index = torch.tensor(np.argwhere(mask).T, dtype=torch.long)
        return Data(
            x=self.local_feat[i],
            edge_index=edge_index,
            global_feat=self.global_feat[i].unsqueeze(0),   # (1, 22)
            y=self.labels[i],
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._graphs[idx]
        if isinstance(idx, (list, tuple)):
            return [self._graphs[i] for i in idx]
        return self._graphs[idx]
