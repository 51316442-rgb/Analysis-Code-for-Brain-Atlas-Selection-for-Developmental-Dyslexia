function extract(atlas_name)

if nargin < 1 || isempty(atlas_name)
    atlas_name = 'aal90';
end
switch lower(atlas_name)
    case 'aal90'
        n_roi = 90;   atlas_file = 'AAL.nii';
    case 'power264'
        n_roi = 264;  atlas_file = 'Power264.nii';
    case 'craddock200'
        n_roi = 200;  atlas_file = 'Craddock200.nii';
    otherwise
        error('unknown atlas: %s (use aal90 / power264 / craddock200)', atlas_name);
end




DATA_ROOT = 'D:/preprocessed_data_root';
OUT_DIR   = 'D:/output_folder';


ATLAS     = fullfile(fileparts(mfilename('fullpath')), 'atlas', atlas_file);
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

WIN       = 3;          % window length (TR)
STEP      = 1;          % step (TR)
SPARS     = 0.05:0.05:0.50;   % GRETNA sparsity levels
WINDOWS_PER_SUBJECT = 80;

% subject order: s1..s33, s7 missing -> 32 subjects
subjects = [1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 ...
            22 23 24 25 26 27 28 29 30 31 32 33];

% Task-block TR ranges (file TR = design TR - 3 dummy removed by DPARSF).
% Each row = [start_TR end_TR] of the 12-TR trials segment.
% s1-s4 paradigm (fAFBFC / FBFCFA / FCFAFB / FAFBFC), A=0-back C=2-back:
%   order [A C C A C A A C] = [0 1 1 0 1 0 0 1]
blocks_s14 = [ 2  13;  40  51;  78  89;  97 108; ...
              116 127; 135 146; 173 184; 211 222];
% s5-s33 paradigm (fAFBFBFA / FAFBFBFA), A=0-back B=2-back:
%   order [A B B A A B B A] = [0 1 1 0 0 1 1 0]
blocks_s5 = [ 2  13;  21  32;  40  51;  59  70; ...
              78  89;  97 108; 116 127; 135 146];

fprintf('=== atlas: %s (%d ROIs) ===\n', atlas_name, n_roi);


network_rows = cell(1, numel(subjects));
feature_rows = cell(1, numel(subjects));
for si = 1:numel(subjects)
    sid = subjects(si);
    if sid <= 4
        sub_dir = fullfile(DATA_ROOT, 's1-s4', 'FunImgARWS', sprintf('s%d', sid));
        blocks  = blocks_s14;
    else
        sub_dir = fullfile(DATA_ROOT, 's5-s33', 'FunImgARWS', sprintf('s%d', sid));
        blocks  = blocks_s5;
    end
    assert(exist(sub_dir, 'dir') == 7, 'subject dir not found: %s', sub_dir);

    files = dir(fullfile(sub_dir, '*.nii'));
    assert(~isempty(files), 'no nii files in %s', sub_dir);
    names = {files.name}';
    num   = cellfun(@(f) str2double(regexp(f, '_(\d+)\.nii$', 'tokens', 'once')), names);
    [~, ord] = sort(num);
    files_full = cellfun(@(f) fullfile(sub_dir, f), names(ord), 'UniformOutput', false);

    fprintf('[%d/%d] subject s%d: %d TRs ... ', si, numel(subjects), sid, numel(files_full));
    ts = read_roi_timeseries(files_full, ATLAS, n_roi);
    r_mats = sliding_corr_blocks(ts, blocks, WIN, STEP);        % (n,n,80)

    n_win = size(r_mats, 3);
    assert(n_win == WINDOWS_PER_SUBJECT, 'unexpected windows: %d', n_win);
    feat = zeros(n_roi, 6, n_win);
    for w = 1:n_win
        feat(:, :, w) = local_features(r_mats(:, :, w), SPARS);
    end

    % flatten row-wise for lossless Python reshape
    nr = zeros(n_win, n_roi * n_roi);
    nf = zeros(n_win, n_roi * 6);
    for w = 1:n_win
        nr(w, :) = reshape(r_mats(:, :, w)', 1, n_roi * n_roi);
        nf(w, :) = reshape(feat(:, :, w)', 1, n_roi * 6);
    end
    network_rows{si} = nr;
    feature_rows{si} = nf;
    fprintf('done.\n');
end

network_flat = cat(1, network_rows{:});
feature_flat = cat(1, feature_rows{:});
fprintf('network_flat: %dx%d, feature_flat: %dx%d\n', ...
        size(network_flat, 1), size(network_flat, 2), ...
        size(feature_flat, 1), size(feature_flat, 2));

save(fullfile(OUT_DIR, sprintf('network_%s.mat', atlas_name)), 'network_flat', '-v7');
save(fullfile(OUT_DIR, sprintf('feature_%s.mat', atlas_name)), 'feature_flat', '-v7');
fprintf('Saved to %s\n', OUT_DIR);
end
