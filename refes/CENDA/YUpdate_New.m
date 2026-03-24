function [Y_new] = YUpdate_New(lower_train_data, Y_ori, k, InsSet)
% YUpdate_New updates the labeling confidence matrix according to the derived lower-dimensional data.
%
% Syntax
%
%       [Y_new] = YUpdate_New(lower_train_data, Y_ori, k, InsSet)
%
% Description
%
%       YUpdate_New takes,
%           lower_train_data    - Derived eigenvalues
%           Y_ori               - The original labeling confidence matrix
%           k                   - The number of the neighboors
%           InsSet              - A M x (Q + 1) matrix, InsSet(i, 1)
%           represent the number of labels regarding ins i and InsSet(i, j + 1) represent the j_th label related to ins i
%
%      and returns,
%           Y_new               - The updated labeling confidence matrix
%

M = size(lower_train_data, 2);
Q = size(Y_ori, 2);

data = lower_train_data';       %   for the arguments' requirement of knn function

%   find knn neighbours
Mdl = KDTreeSearcher(data); 
[Idx] = knnsearch(Mdl, data, 'k', k + 1);
Idx = Idx(:, 2:end);

%   calculate matrix Z and V
Z = zeros(M, Q);
V = zeros(M, Q);
for i = 1:M
    neighbours = Idx(i, :);
    
    for nei = 1:k
        ind = neighbours(nei);
        
        nLabels = InsSet(ind, 1);
        Labels = InsSet(ind, 2:(nLabels + 1));
        
        for it = 1:nLabels
            j = Labels(it);
            Z(i, j) = Z(i, j) + Y_ori(ind, j) * (k - nei + 1);
            V(i, j) = V(i, j) + 1;
        end
        
    end
end


Z_New = Y_ori;
for i = 1:M
    nLabels = InsSet(i, 1);
    Labels = InsSet(i, 2:(nLabels + 1));
    
    for it = 1:nLabels
        j = Labels(it);
        Z_New(i, j) = Z_New(i, j) + Z(i, j);
    end
    
    if sum(Z_New(i, :)) ~= 0
        Z_New(i, :) = Z_New(i, :) ./ sum(Z_New(i, :));
    else
        Z_New(i, :) = Y_ori(i, :);
    end
end

Y_new = Z_New;

end
