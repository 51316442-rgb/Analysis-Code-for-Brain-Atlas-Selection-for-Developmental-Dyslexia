function ts = read_roi_timeseries(nii_files, atlas_path, n_roi)
% read_roi_timeseries  Extract ROI mean time series from preprocessed 3D nii volumes.

if ischar(nii_files) || isstring(nii_files)
    nii_files = {char(nii_files)};
end


func_info = niftiinfo(nii_files{1});
labels = resample_atlas_to_func(atlas_path, func_info);

uv = unique(labels(labels > 0));
assert(numel(uv) >= n_roi, 'atlas has only %d regions, need %d', numel(uv), n_roi);
roi_values = uv(1:n_roi);
roi_mask = cell(n_roi, 1);
for k = 1:n_roi
    roi_mask{k} = (labels == roi_values(k));
end


first = niftiread(nii_files{1});
if ndims(first) == 4
    n_tr = size(first, 4);
    all4d = true;
else
    n_tr = numel(nii_files);
    all4d = false;
end

ts = zeros(n_tr, n_roi, 'double');
tr = 0;
for f = 1:numel(nii_files)
    img = niftiread(nii_files{f});
    if all4d
        for t = 1:size(img, 4)
            tr = tr + 1;
            vol = double(img(:, :, :, t));
            for k = 1:n_roi
                m = roi_mask{k};
                vals = vol(m);
                vals = vals(isfinite(vals));
                ts(tr, k) = mean(vals);
            end
        end
    else
        tr = tr + 1;
        vol = double(img);
        for k = 1:n_roi
            m = roi_mask{k};
            vals = vol(m);
            vals = vals(isfinite(vals));
            ts(tr, k) = mean(vals);
        end
    end
end


empty_roi = find(all(ts == 0, 1) | any(isnan(ts), 1));
if ~isempty(empty_roi)
    warning('read_roi_timeseries:emptyROI', 'ROIs with no valid voxels: %s', mat2str(empty_roi));
end
end
