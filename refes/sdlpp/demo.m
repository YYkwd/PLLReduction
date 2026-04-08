clear;
clc;

% Project root: .../PllReduction/refes/sdlpp/demo.m -> go up twice to refes, once more to root
this_file = mfilename('fullpath');
if isempty(this_file)
    error('Save this file as demo.m and run from MATLAB, or set data_file manually.');
end
sdlpp_dir = fileparts(this_file);
refes_dir = fileparts(sdlpp_dir);
proj_root = fileparts(refes_dir);
data_file = fullfile(proj_root, 'datasets', 'Soccer Player.mat');

if exist(data_file, 'file') ~= 2
    error('Dataset not found: %s\nEdit data_file in demo.m if your layout differs.', data_file);
end

load(data_file);

if issparse(partial_target)
    partial_target = full(partial_target);
end
if exist('target', 'var') && issparse(target)
    target = full(target);
end

if ~exist('target', 'var')
    error('Dataset must contain ''target'' (ground-truth one-hot, Q x M) for evaluation.');
end

[~, y_true] = max(target, [], 1);
y_true = y_true(:);

% Hyper-parameters (match configs/default.yaml style)
para.T = 100;
para.target_d = 13;
para.k = 8;
para.miu = 0.1;
para.thr = 0.95;

knn_k = 5;
cv_folds = 5;
rng(42);

C = cvpartition(y_true, 'KFold', cv_folds);

acc = zeros(cv_folds, 1);
bal_acc = zeros(cv_folds, 1);

for fold = 1:cv_folds
    tr = training(C, fold);
    te = test(C, fold);

    X_tr = data(tr, :);
    X_te = data(te, :);
    pt_tr = partial_target(:, tr);

    % Z-score: fit on train only (ddof N-1, same as MATLAB zscore / Python ddof=1)
    mu = mean(X_tr, 1);
    sig = std(X_tr, 0, 1);
    sig(sig == 0) = 1;
    X_tr_z = (X_tr - mu) ./ sig;
    X_te_z = (X_te - mu) ./ sig;

    [Z_tr, P] = SDLPP(X_tr_z, pt_tr, para);
    Z_te = X_te_z * P;

    y_tr = y_true(tr);
    y_te = y_true(te);

    % Manual kNN (Euclidean + majority vote). fitcknn/predict can error when
    % ClassificationKNN calls gather() and a different gather shadows MATLAB's.
    y_hat = knn_predict_euclidean(Z_tr, y_tr, Z_te, knn_k);

    acc(fold) = mean(y_hat == y_te);
    bal_acc(fold) = balanced_acc(y_te, y_hat);
end

fprintf('\n========== SDLPP + KNN (downstream) ==========\n');
fprintf('Dataset: %s\n', data_file);
fprintf('CV folds: %d, KNN k: %d\n', cv_folds, knn_k);
fprintf('Overall accuracy:  mean = %.4f, std = %.4f\n', mean(acc), std(acc));
fprintf('Balanced accuracy: mean = %.4f, std = %.4f\n', mean(bal_acc), std(bal_acc));
fprintf('==============================================\n\n');

function y_hat = knn_predict_euclidean(Z_tr, y_tr, Z_te, k)
% kNN classification: Euclidean distance, same k neighbors as sklearn KNeighborsClassifier.
    n_te = size(Z_te, 1);
    n_tr = size(Z_tr, 1);
    y_hat = zeros(n_te, 1);
    y_tr = y_tr(:);
    k = min(k, n_tr);
    for i = 1:n_te
        d = sum(bsxfun(@minus, Z_tr, Z_te(i, :)).^2, 2);
        [~, ix] = sort(d);
        nn = ix(1:k);
        y_hat(i) = mode(y_tr(nn));
    end
end

function ba = balanced_acc(y_true, y_pred)
% Average per-class recall (same spirit as sklearn balanced_accuracy_score).
    y_true = y_true(:);
    y_pred = y_pred(:);
    labels = unique(y_true);
    n = numel(labels);
    rec = zeros(n, 1);
    for i = 1:n
        c = labels(i);
        m = (y_true == c);
        if any(m)
            rec(i) = sum(y_pred(m) == c) / sum(m);
        else
            rec(i) = 0;
        end
    end
    ba = mean(rec);
end
