import os

import numpy as np
from scipy.io import savemat
from scipy.sparse.csgraph import shortest_path

ROI = 90
MODULES = 6
N_SUBJECTS = 12
WINDOWS_PER_SUBJECT = 20
SPARSITIES = np.arange(0.05, 0.51, 0.05)

DD_SUBJECTS = list(range(1, 7))          # subjects 1-6 = DD, 7-12 = Control
DD_RHO = (0.45, 0.30)                    # weak modular structure
CTL_RHO = (0.85, 0.02)                   # strong modular structure
DD_MIX_PROB = 0.35                       # prob. a DD window is drawn from the
                                         # Control structure (window heterogeneity)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def gen_networks(rho_in, rho_out, n_windows, roi, modules, rng):
    per_mod = roi // modules
    mod = np.repeat(np.arange(modules), per_mod)
    mats = []
    for _ in range(n_windows):
        R = np.full((roi, roi), rho_out)
        for m in range(modules):
            idx = np.where(mod == m)[0]
            R[np.ix_(idx, idx)] = rho_in
        noise = rng.normal(0, 0.05, (roi, roi))
        R = R + noise
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 1.0)
        mats.append(R)
    return np.stack(mats)


def gen_networks_mix(rho_in_dd, rho_out_dd, rho_in_ctl, rho_out_ctl,
                     n_windows, roi, modules, rng, mix_prob):
    per_mod = roi // modules
    mod = np.repeat(np.arange(modules), per_mod)
    mats = []
    for _ in range(n_windows):
        if rng.random() < mix_prob:
            rho_in, rho_out = rho_in_ctl, rho_out_ctl
        else:
            rho_in, rho_out = rho_in_dd, rho_out_dd
        R = np.full((roi, roi), rho_out)
        for m in range(modules):
            idx = np.where(mod == m)[0]
            R[np.ix_(idx, idx)] = rho_in
        noise = rng.normal(0, 0.05, (roi, roi))
        R = R + noise
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 1.0)
        mats.append(R)
    return np.stack(mats)


def _betweenness(adjlist, n):
    bc = np.zeros(n)
    for s in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        sigma = np.zeros(n, dtype=np.float64)
        dist[s] = 0
        sigma[s] = 1.0
        stack = []
        q = [s]
        head = 0
        while head < len(q):
            v = q[head]
            head += 1
            stack.append(v)
            dv = dist[v] + 1
            for w in adjlist[v]:
                if dist[w] < 0:
                    dist[w] = dv
                    q.append(w)
                if dist[w] == dv:
                    sigma[w] += sigma[v]
        delta = np.zeros(n)
        for v in reversed(stack):
            sv = sigma[v]
            for w in adjlist[v]:
                if dist[w] == dist[v] - 1:
                    delta[w] += (sigma[w] / sv) * (1.0 + delta[v])
            if v != s:
                bc[v] += delta[v]
    return bc / ((n - 1) * (n - 2))


def graph_metrics(A):
    n = A.shape[0]
    adjlist = [np.nonzero(A[i])[0].tolist() for i in range(n)]
    deg = A.sum(axis=1)/ (n - 1)
    bc = _betweenness(adjlist, n)
    k = deg.astype(float)
    T = np.diag(A @ A @ A) / 2.0
    denom = k * (k - 1)
    clu = np.where(denom > 0, 2.0 * T / np.maximum(denom, 1e-12), 0.0)
    D = shortest_path(A, method="D", directed=False, unweighted=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(D == 0, 0.0, 1.0 / D)
    eglob = inv.sum(axis=1) / max(n - 1, 1)
    eloc = np.zeros(n)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        kk = len(nbrs)
        if kk > 1:
            Ds = shortest_path(A[np.ix_(nbrs, nbrs)], method="D",
                               directed=False, unweighted=True)
            invs = np.where(Ds == 0, 0.0, 1.0 / Ds)
            eloc[i] = invs.sum() / (kk * (kk - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        Df = np.where(D == 0, np.inf, D)
        lp = np.nanmean(np.where(np.isfinite(Df), Df, np.nan), axis=1)
        lp = np.where(np.isnan(lp), 0.0, lp)
    return np.column_stack([deg, bc, clu, eglob, eloc, lp])


def local_features(r, sparsities):
    n = r.shape[0]
    iu = np.triu_indices(n, k=1)
    w = np.abs(r[iu])
    feats = np.zeros((n, 6))
    for s in sparsities:
        k = max(1, int(round(s * w.size)))
        th = np.percentile(w, 100 * (1 - s))
        A = (np.abs(r) > th).astype(float)
        np.fill_diagonal(A, 0.0)
        feats += graph_metrics(A)
    return feats / sparsities.size


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)
    mats = []
    for s in range(1, N_SUBJECTS + 1):
        if s in DD_SUBJECTS:
            mats.append(gen_networks_mix(DD_RHO[0], DD_RHO[1],
                                         CTL_RHO[0], CTL_RHO[1],
                                         WINDOWS_PER_SUBJECT, ROI, MODULES,
                                         rng, DD_MIX_PROB))
        else:
            mats.append(gen_networks(CTL_RHO[0], CTL_RHO[1],
                                     WINDOWS_PER_SUBJECT, ROI, MODULES, rng))
    network = np.concatenate(mats, axis=0).astype(np.float32)   # (240,90,90)
    feats = np.stack([local_features(network[w], SPARSITIES)
                      for w in range(network.shape[0])]).astype(np.float32)

    np.save(os.path.join(OUT_DIR, "network_synthetic.npy"), network)
    savemat(os.path.join(OUT_DIR, "feature_synthetic.mat"),
            {"feature_flat": feats.reshape(feats.shape[0], -1)})

    with open(os.path.join(OUT_DIR, "labels_synthetic.csv"), "w") as f:
        f.write("subject,group\n")
        for s in range(1, N_SUBJECTS + 1):
            f.write(f"{s},{'DD' if s in DD_SUBJECTS else 'Control'}\n")

    print("saved:")
    print("  network_synthetic.npy ", network.shape)
    print("  feature_synthetic.mat ", feats.shape, "(flat", feats.reshape(feats.shape[0], -1).shape, ")")
    print("  labels_synthetic.csv  ", N_SUBJECTS, "subjects, DD =", DD_SUBJECTS)


if __name__ == "__main__":
    main()
