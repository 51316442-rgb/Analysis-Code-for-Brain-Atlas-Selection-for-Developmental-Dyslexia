# Analysis Code for Brain Atlas Selection for Developmental Dyslexia

Analysis code for classifying developmental dyslexia (DD) versus control subjects from fMRI data, using graph-theoretical and topological features fused with an attention mechanism and classified by a two-layer GCN.

Requires MATLAB and Python 3 (numpy, scipy, pandas, torch, torch-geometric, scikit-learn, xgboost, giotto-tda).

MATLAB scripts in the root directory extract brain networks and local graph-theoretic features from preprocessed fMRI time series, supporting three atlases: AAL90, Power264, and Craddock200.

Python scripts in the root directory perform attention-fused GCN classification with subject-wise cross-validation.

Scripts in the ph folder compute persistent-homology (PH) topological features per window and related utilities for point-cloud construction, persistence intervals, merging.

The atlas folder contains the three brain atlases used in the paper: AAL (90 ROIs), Power264 (264 ROIs), and Craddock200 (200 ROIs).

## Simulated data

Due to ethical review constraints imposed on human‑subject neuroimaging experiments, individual‑level fMRI and real functional‑connectivity matrices cannot be publicly deposited. This small synthetic test set (12 subjects: 1-6 DD, 7-12 control; 20 windows per person; 90 ROIs; control group has clear modular structure, DD group has uniform and weak connections) shows how the whole process runs from start to finish. It's not real data; the results are just for demonstration and don't represent any real findings.

run the example:

```
python train.py --network data\network_synthetic.npy --local data\feature_synthetic.mat --ph data\fMRI_features_synthetic.csv --dys-subjects "1,2,3,4,5,6" --windows-per-subject 20 --threshold 0.1 --epochs 6 --batch 64 --seed 0
```

- `data\network_synthetic.npy` — synthetic functional-connectivity matrices, one 90×90 matrix per window (240 windows total).
- `data\feature_synthetic.mat` — local graph-theoretic features per window: 6 node metrics (degree, betweenness, clustering, global/local efficiency, average path length) for each of the 90 ROIs.
- `data\fMRI_features_synthetic.csv` — PH topological features per window (22 columns: persistence entropy plus landscape / Betti / heat-kernel amplitudes).
- `data\labels_synthetic.csv` — subject grouping labels (12 subjects: 1-6 DD, 7-12 Control).







