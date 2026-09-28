function r_mats = sliding_corr_blocks(ts, block_ranges, win, step)

n_blocks = size(block_ranges, 1);
n_roi = size(ts, 2);
n_win_block = floor((block_ranges(1, 2) - block_ranges(1, 1) + 1 - win) / step) + 1;
r_mats = zeros(n_roi, n_roi, n_blocks * n_win_block);

w = 0;
for b = 1:n_blocks
    t0 = block_ranges(b, 1);
    t1 = block_ranges(b, 2);
    for t = t0:step:(t1 - win + 1)
        w = w + 1;
        seg = ts(t:t + win - 1, :);          % (win, n_roi)
        r = corrcoef(seg);                    % (n_roi, n_roi)
        if size(r, 1) ~= n_roi
            % corrcoef drops constant columns -> rebuild with explicit pairwise
            r = nan(n_roi);
            for i = 1:n_roi
                for j = 1:n_roi
                    r(i, j) = corr(seg(:, i), seg(:, j));
                end
            end
        end
        r_mats(:, :, w) = r;
    end
end
end
