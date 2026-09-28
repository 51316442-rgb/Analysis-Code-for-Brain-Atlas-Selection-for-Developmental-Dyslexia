function feats = local_features(r, sparsities)
% local_features  GRETNA-style local graph-theoretic features.
%   For each sparsity level, binarize the correlation matrix and average
%   the 6 nodal metrics (DC/BC/C/E_glob/E_loc/L_p) over all levels.
n = size(r, 1);
iu = find(triu(true(n), 1));
w = abs(r(iu));
feats = zeros(n, 6);
for s = sparsities
    k = max(1, round(s * numel(w)));
    th = prctile(w, 100 * (1 - s));
    A = double(abs(r) > th);
    A(1:n + 1:end) = 0;
    feats = feats + graph_metrics(A);
end
feats = feats / numel(sparsities);
end
