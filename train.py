import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier

from model import BrainGCNClassifier
from dataset import BrainGraphDataset

N_FOLDS = 4
GRAPH_THRESHOLD = 0.0


DYS_SUBJECTS = {2, 3, 16, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_labels(mode, subjects, dys_subjects):
    uniq = np.sort(np.unique(subjects))
    n_subjects = len(uniq)
    if mode == "real":
        group = {s: (1 if (s + 1) in dys_subjects else 0) for s in uniq}
    else:
        half = n_subjects // 2
        group = {s: (0 if i < half else 1) for i, s in enumerate(uniq)}
    return np.array([group[s] for s in subjects])


def load_features(net_path, local_path, ph_path, n_roi):
    import pandas as pd

    if not os.path.exists(ph_path):
        raise FileNotFoundError(f"missing PH feature csv: {ph_path}")
    df = pd.read_csv(ph_path, header=0)
    x_ph = df.iloc[:, 1:].values.astype(np.float32)
    N = x_ph.shape[0]

    def load_array(path, shape):
        if not os.path.exists(path):
            return None
        if path.lower().endswith(".npy"):
            arr = np.load(path)
        else:
            m = loadmat(path)
            key = [k for k in m if not k.startswith("__")][0]
            arr = np.asarray(m[key])
        return arr.reshape(*shape).astype(np.float32)

    x_local = load_array(local_path, (N, n_roi, 6))
    adj = load_array(net_path, (N, n_roi, n_roi))

    if x_local is None or adj is None:
        missing = [p for p, v in zip(["network", "local features"],
                                     [adj is not None, x_local is not None]) if not v]
        raise FileNotFoundError(
            f"missing input file(s): {', '.join(missing)}. "
            f"Pass --network / --local / --ph with existing paths "
            f"(see README).")
    return x_local, x_ph, adj


def zscore_fit_transform(x, idx):
    x_tr = x[idx]
    flat = x_tr.reshape(-1, x_tr.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def select_top10(x_ph, y, seed):
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                        subsample=0.8, random_state=seed, n_jobs=1)
    clf.fit(x_ph, y)
    return np.argsort(clf.feature_importances_)[::-1][:10]


def evaluate(model, loader, device, clf_thr=0.5):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            ys.extend(data.y.cpu().tolist())
            preds.extend((torch.softmax(out, dim=1)[:, 1] >= clf_thr)
                         .long().cpu().tolist())
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().tolist())
    return {
        "acc": accuracy_score(ys, preds),
        "precision": precision_score(ys, preds, zero_division=0),
        "recall": recall_score(ys, preds, zero_division=0),
        "f1": f1_score(ys, preds, zero_division=0),
        "auc": roc_auc_score(ys, probs),
    }, ys, preds, probs


def run_fold(x_local, x_ph, adj, labels, subjects, tr_idx, va_idx,
             d_embed, seed, epochs, batch_size, threshold, clf_thr=0.5,
             pos_weight=1.0):
    sel = select_top10(x_ph[tr_idx], labels[tr_idx], seed)
    x_local_n = zscore_fit_transform(x_local, tr_idx)
    x_ph_n = zscore_fit_transform(x_ph[:, sel], tr_idx)

    tr_ds = BrainGraphDataset(x_local_n[tr_idx], x_ph_n[tr_idx],
                              adj[tr_idx], labels[tr_idx], threshold=threshold)
    va_ds = BrainGraphDataset(x_local_n[va_idx], x_ph_n[va_idx],
                              adj[va_idx], labels[va_idx], threshold=threshold)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainGCNClassifier(local_dim=6, global_dim=len(sel),
                               d_embed=d_embed, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    w = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data in tr_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == data.y).sum().item()
            total += data.num_graphs
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}  train_acc={correct / total:.4f} "
                  f"loss={total_loss / len(tr_loader):.4f}")

    metrics, _, _, _ = evaluate(model, va_loader, device, clf_thr)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True,
                    help="network matrices .mat from MATLAB step (N, n_roi^2 flat)")
    ap.add_argument("--local", required=True,
                    help="local graph features .mat from MATLAB step (N, n_roi*6 flat)")
    ap.add_argument("--ph", required=True,
                    help="PH feature CSV from step 2 (N rows, 1st col subject id)")
    ap.add_argument("--n-roi", type=int, default=90,
                    help="number of ROIs (90 / 264 / 200)")
    ap.add_argument("--windows-per-subject", type=int, default=80,
                    help="windows per subject, used to build subject groups "
                         "(80 for the full dataset, 20 for the synthetic example)")
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--labels", choices=["real", "old"], default="real")
    ap.add_argument("--dys-subjects", default=",".join(str(s) for s in sorted(DYS_SUBJECTS)),
                    help="comma-separated 1-based subject indices in the DD group "
                         "(used by --labels real)")
    ap.add_argument("--threshold", type=float, default=GRAPH_THRESHOLD)
    ap.add_argument("--clf-thr", type=float, default=0.5,
                    help="decision threshold for the DD class (0.5 = argmax; "
                         "raise it to trade recall for precision)")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="loss weight for the DD class (<1.0 makes the model "
                         "less eager to predict DD, lowering recall)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    n_roi = args.n_roi
    x_local, x_ph, adj = load_features(args.network, args.local, args.ph, n_roi)
    dys_subjects = {int(s) for s in args.dys_subjects.split(",") if s.strip()}
    N = x_ph.shape[0]
    subjects = np.arange(N) // args.windows_per_subject
    labels = build_labels(args.labels, subjects, dys_subjects)
    n_subjects = len(np.unique(subjects))

    print("=" * 72)
    print("Paper pipeline step 3: attention fusion + GCN (subject-wise CV)")
    print(f"n_roi={n_roi}  labels={args.labels}  d={args.d}  "
          f"epochs={args.epochs}  threshold={args.threshold}  seed={args.seed}")
    print(f"X_local {x_local.shape}  X_ph {x_ph.shape}  adj {adj.shape}")
    print(f"groups: {n_subjects} subjects x "
          f"{len(subjects) // n_subjects} windows "
          f"(dys windows={labels.sum()})")
    print("=" * 72)

    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x_local, labels, subjects), 1):
        print(f"---- Fold {fold}/{N_FOLDS}  train {len(tr_idx)} / val {len(va_idx)} ----")
        metrics = run_fold(x_local, x_ph, adj, labels, subjects, tr_idx, va_idx,
                           args.d, args.seed, args.epochs, args.batch,
                           args.threshold, args.clf_thr, args.pos_weight)
        print(f"  Val: acc={metrics['acc']:.4f} p={metrics['precision']:.4f} "
              f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
        rows.append(metrics)

    print("\n" + "=" * 72)
    print(f"subject-wise CV summary (n_roi={n_roi}, d={args.d}, "
          f"labels={args.labels}, seed={args.seed})")
    for k in ("acc", "precision", "recall", "f1", "auc"):
        v = [r[k] for r in rows]
        print(f"  {k:10s}: {np.mean(v):.4f} +- {np.std(v):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
