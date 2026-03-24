function [projection_matrix, lower_train_data] = CENDA(train_data, train_p_target, T, mu, dim_para, k)
% CENDA [1] deals with partial label dimensionality reduction which adapts the Hilbert-Schmidt Independence Criterion (HSIC) to maximize the dependence between projected feature information and confidence-based labeling information iteratively.
%
% Syntax
%
%       [projection_matrix, lower_train_data] = CENDA(data, partial_target, T, mu, dim_para, k)
%
% Description
%
%       DELIN takes,
%           train_data          - A M x D array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target      - A Q x M array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           T                   - The number of iterations
%           mu                  - The trade-off parameter which balances the orthonormal constraints over projection bases and the uncorrelatedness constraints over projected features
%           dim_para            - The threshold which controls the dimensionality of the projected feature space
%           k                   - The number of the neighboors
%      
%       and returns,
%           projection_matrix     - An D x D' array, D' is the number of retained features,the projection matrix
%           lower_train_data      - An M x D' array, a projected data matrix
%          
%
%  [1] W.-X. Bao, J.-Y. Hang, M.-L. Zhang. Partial Label Dimensionality Reduction via Confidence-Based Dependence Maximization (KDD'21), Virtual Event, Singapore, 2021, in press.
%

D = size(train_data, 2);                %   Dimension of original data
M = size(train_data, 1);                %   Number of instances
Q = size(train_p_target, 1);            %   Number of labels

X = zscore(train_data);
X = X';

LabelSet = zeros(Q, M + 6);     %   LabelSet(i, 1) represent the number of instances regarding label i and LableSet(i, j + 1) represent the index of j_th instance
InsSet = zeros(M, Q + 6);       %   InsSet(i, 1) represent the number of labels regarding ins i and InsSet(i, j + 1) represent the j_th label
%   Initialize LabelSet and InsSet
train_p_target = train_p_target';
for i = 1:M
    for j = 1:Q
        if train_p_target(i, j) == 1
            LabelSet(j, 1) = LabelSet(j, 1) + 1;
            t = LabelSet(j, 1);
            LabelSet(j, t + 1) = i;
            
            InsSet(i, 1) = InsSet(i, 1) + 1;
            t = InsSet(i, 1);
            InsSet(i, t + 1) = j;
        end
    end
end

%   Initialize label confidence matrix Y
Y = zeros(M, Q);
for i = 1:M
    tot = InsSet(i, 1);
    if tot ~= 0
        for j = 1:tot
            la = InsSet(i, j + 1);
            Y(i, la) = 1 / tot;
        end
    end
    
    Y(i, :) = Y(i, :) / norm(Y(i, :), 1);
    
end


for t = 1:T
   
    L = Y * Y';
    
    [P, lambda] = HSIC_Based_Solver(X, L, mu, dim_para);
    lower_train_data = P' * X;
    
    [Y_new] = YUpdate_New(lower_train_data, Y, k, InsSet);
    Y = Y_new;
end

projection_matrix = P;

end
