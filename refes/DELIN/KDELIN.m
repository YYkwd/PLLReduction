function [Theta, lower_train_data] = KDELIN(train_data, train_p_target, ratio, k, T)

%   Function KDELIN
%   Here is a brief introduction:
%   
%   Inputs:
%   train_data: an D*M array; D is the number of features and M is the number of instances
%   tain_p_target: an M*Q arry; Q is the number of labels
%   ratio: the threshold of retained features
%   k: the number of neighbours
%   T: the number of iterations
%   
%   Outputs:
%   Theta: an D*d array; d is the reduced dimension
%   lower_train_data: an d*M array

D = size(train_data, 1);        %   dimension of original data
M = size(train_data, 2);        %   number of instances
Q = size(train_p_target, 2);    %   number of labels

LabelSet = zeros(Q, M + 6);     %   LabelSet(i, 1) represent the number of instances regarding label i and LableSet(i, j + 1) represent the index of j_th instance
InsSet = zeros(M, Q + 6);       %   InsSet(i, 1) represent the number of labels regarding ins i and InsSet(i, j + 1) represent the j_th label
%   Initialize LabelSet and InsSet
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
end

%   get kernel matrix
K = getKernel(train_data, train_data);

%   start to iterate
for t = 1:T
    [Theta, lower_train_data] = Dimen_Red(K, Y, ratio, LabelSet);
    
    if t == T
        break;
    end
    
    [Y] = YUpdate(lower_train_data, Y, k, InsSet);
    
end

end
