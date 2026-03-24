function [P lambda] = HSIC_Based_Solver(X, L, mu, dim_para)
% HSIC_Based_Solver identifies the projected matrix by solving a tailored generalized eigenvalue problem derived from the adapted HSIC
%
% Syntax
%
%       [P lambda] = HSIC_Based_Solver(X, L, mu, dim_para)
%
% Description
%
%       HSIC_Based_Solver takes,
%           X                   - A D x M array, the ith instance of training instance is stored in X(:,i)
%           L                   - A M x M array, the kernel matrix for label space
%           mu                  - The trade-off parameter which balances the orthonormal constraints over projection bases and the uncorrelatedness constraints over projected features
%           dim_para            - The threshold which controls the dimensionality of the projected feature space
%      
%       and returns,
%           P                   - An D x D' array, D' is the number of retained features,the projection matrix
%           lambda              - Derived eigenvalues
%          
%

[D N] = size(X);
tmpL = L - repmat(mean(L, 1), N, 1);
HLH = tmpL - repmat(mean(tmpL, 2), 1, N);

S = X * HLH * X';

B = mu * X * X' + (1 - mu) * eye(D);

clear X L;

[tmp_P, tmp_lambda] = eig(S, B);
tmp_P = real(tmp_P);
tmp_lambda = real(diag(tmp_lambda));
[lambda, order] = sort(tmp_lambda, 'descend');
P = tmp_P(:,order);

proper_dim = getProperDim(lambda, dim_para);
P = P(:,1:proper_dim);
lambda = lambda(1:proper_dim);

end

