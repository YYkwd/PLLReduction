function [Y_new] = YUpdate(lower_train_data, Y_ori, k, InsSet)
%   Function YUPDATE

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

%   find L_j_star
L_j_star = zeros(M, 1);
for i = 1:M
    nLabels = InsSet(i, 1);
    Labels = InsSet(i, 2:(nLabels + 1));
    
    extract = Z(i, Labels);
    [max_z] = max(extract);
    index = find(extract == max_z);
    
    num = size(index, 2);
    randind = randi(num);
    
    L_j_star(i, 1) = Labels(randind);
    
end

%   update Y
Y_new = zeros(M, Q);
for i = 1:M
    nLabels = InsSet(i, 1);
    Labels = InsSet(i, 2:(nLabels + 1));
    j_star = L_j_star(i, 1);
    
    if nLabels == 1
        Y_new(i, j_star) = 1;
    else
        lambda = floor(k / 2);      % parameter to avoid V_ij / k == 1
        Y_new(i, j_star) = V(i, j_star) / (k + lambda);
        for it = 1:nLabels
            j = Labels(it);
            if j ~= j_star
                Y_new(i, j) = (1 - V(i, j_star) / (k + lambda)) / (nLabels - 1);
            end
        end
    end
    
end

end
