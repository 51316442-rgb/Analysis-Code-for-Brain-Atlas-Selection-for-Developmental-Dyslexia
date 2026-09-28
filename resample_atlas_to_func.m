function labels_res = resample_atlas_to_func(atlas_path, func_info)
% resample_atlas_to_func  Nearest-neighbor resample of an atlas

A = double(niftiread(atlas_path));
infoA = niftiinfo(atlas_path);
T_A = infoA.Transform.T;
T_F = func_info.Transform.T;
[nx, ny, nz] = size(A);
f = niftiread(func_info.Filename);
[mx, my, mz] = size(f);

[x, y, z] = ndgrid(1:mx, 1:my, 1:mz);
world = [x(:), y(:), z(:), ones(numel(x), 1)] * T_F;
world = world(:, 1:3);

idx = [world, ones(size(world, 1), 1)] * inv(T_A);
ia = round(idx(:, 1));
ja = round(idx(:, 2));
ka = round(idx(:, 3));
ok = ia >= 1 & ia <= nx & ja >= 1 & ja <= ny & ka >= 1 & ka <= nz;

labels_res = zeros(mx, my, mz, 'int16');
lin = sub2ind([nx, ny, nz], ia(ok), ja(ok), ka(ok));
labels_res(ok) = A(lin);
end
