function feats = graph_metrics(A)
% graph_metrics  Six node-level graph-theoretic metrics on a binary undirected graph.
%   Input:
%     A : (n, n) binary adjacency matrix (0/1, symmetric, diagonal ignored)
%   Output:
%     feats : (n, 6) columns =
%       1 degree centrality DC, 2 betweenness centrality BC,
%       3 clustering coefficient C, 4 nodal global efficiency E_glob,
%       5 nodal local efficiency E_loc, 6 average shortest path length L_p

n = size(A, 1);
A = double(A > 0);
A(1:n + 1:end) = 0;


deg = sum(A, 2);
feats(:, 1) = deg / (n - 1);


dist = inf(n);
npaths = zeros(n);
sigma = zeros(n, n);
for s = 1:n
    d = inf(1, n); d(s) = 0;
    sigma_s = zeros(1, n); sigma_s(s) = 1;
    Q = zeros(1, n); qh = 1; qt = 1; Q(qh) = s;
    while qh <= qt
        v = Q(qh); qh = qh + 1;
        nbrs = find(A(v, :));
        for u = nbrs
            if d(u) == inf
                d(u) = d(v) + 1;
                qt = qt + 1; Q(qt) = u;
            end
            if d(u) == d(v) + 1
                sigma_s(u) = sigma_s(u) + sigma_s(v);
            end
        end
    end
    dist(s, :) = d;
    sigma(s, :) = sigma_s;
end


inv_d = 1 ./ dist;
inv_d(dist == 0) = 0;
inv_d(isinf(dist)) = 0;
feats(:, 4) = sum(inv_d, 2) / (n - 1);
finite_d = dist;
finite_d(1:n + 1:end) = nan;
finite_d(isinf(finite_d)) = nan;
Lp = nanmean(finite_d, 2);
Lp(isnan(Lp)) = 0;
feats(:, 6) = Lp;


CB = zeros(n, 1);
for s = 1:n
    S = zeros(1, n); nS = 0;
    P = cell(n, 1);
    d = inf(1, n); d(s) = 0;
    sigma_s = zeros(1, n); sigma_s(s) = 1;
    Q = zeros(1, n); qh = 1; qt = 1; Q(qh) = s;
    while qh <= qt
        v = Q(qh); qh = qh + 1;
        nS = nS + 1; S(nS) = v;
        nbrs = find(A(v, :));
        for u = nbrs
            if d(u) == inf
                d(u) = d(v) + 1;
                qt = qt + 1; Q(qt) = u;
            end
            if d(u) == d(v) + 1
                sigma_s(u) = sigma_s(u) + sigma_s(v);
                P{u}(end + 1) = v;
            end
        end
    end
    delta = zeros(1, n);
    while nS > 0
        w = S(nS); nS = nS - 1;
        for v = P{w}
            delta(v) = delta(v) + (sigma_s(v) / sigma_s(w)) * (1 + delta(w));
        end
        if w ~= s
            CB(w) = CB(w) + delta(w);
        end
    end
end
if n > 2
    feats(:, 2) = CB / ((n - 1) * (n - 2));
else
    feats(:, 2) = 0;
end


for i = 1:n
    nbrs = find(A(i, :));
    k = numel(nbrs);
    if k < 2
        feats(i, 3) = 0;
    else
        sub = A(nbrs, nbrs);
        feats(i, 3) = sum(sub(:)) / (k * (k - 1));
    end
end


for i = 1:n
    nbrs = find(A(i, :));
    if numel(nbrs) < 2
        feats(i, 5) = 0;
    else
        sub = A(nbrs, nbrs);
        m = numel(nbrs);
        dsub = inf(m);
        for s = 1:m
            d = inf(1, m); d(s) = 0;
            Q = zeros(1, m); qh = 1; qt = 1; Q(qh) = s;
            while qh <= qt
                v = Q(qh); qh = qh + 1;
                for u = find(sub(v, :))
                    if d(u) == inf
                        d(u) = d(v) + 1;
                        qt = qt + 1; Q(qt) = u;
                    end
                end
            end
            dsub(s, :) = d;
        end
        inv_d = 1 ./ dsub;
        inv_d(dsub == 0) = 0;
        inv_d(isinf(dsub)) = 0;
        feats(i, 5) = sum(inv_d(:)) / (m * (m - 1));
    end
end
end
