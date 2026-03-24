function [P] = LDA_calculation(train_data ,ratio,Yconfidence)
% LDA_calculation uses LDA framework to calculate projection matrix.
%
% Syntax
%
%       [P] = LDA_calculation(X,ratio,Yconfidence)
%
% Description
%
%       LDA_calculation takes,
%           train_data          - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           ratio               - the threshold of retained features
%           Yconfidence         - An MxQ array, Yconfidence(i,j) denotes the confidence of the jth class label as its true label in the ith instance         
%      
%       and returns,
%           P                   - An NxD array, D is the number of retained features,the LDA projection matrix
%

train_data=train_data';
[N,M] = size(train_data); % N--the number of features
Q = size(Yconfidence,2); % Q--the number of labels
m1=0;
for j=1:Q
    for i=1:M
        m1=m1+Yconfidence(i,j)*train_data(:,i);
    end
end
sumYconfidence=0;
for j=1:Q
    for i=1:M
        sumYconfidence=sumYconfidence+Yconfidence(i,j);
    end
end
m=m1/sumYconfidence;
e=ones(1,M)';
Xm=train_data-m*e';
% the sum of weight for each label and the total sum of weights
% Nk -- a Q-dimensonal row vector
Wk = sum(Yconfidence);
Wt = sum(Wk);

%N1=1./Nk;
for i=1:Q
    if(Wk(i)==0)
        W1(i)=0;
    else
        W1(i) = 1/Wk(i);
    end
end

sumL = sum(Yconfidence');
Ln=diag(sumL);
Hb = Yconfidence*diag(W1)*Yconfidence';
St=Xm*Ln*Xm';
Sb=Xm*Hb*Xm';
Sw=St-Sb;

Sw = (Sw + Sw')/2.0;
Sb = (Sb + Sb')/2.0;
%maxSw = max(max(Sw));
%minSw = min(min(Sw));
maxSw = max(diag(Sw));
minSw = min(diag(Sw));
disp(strcat('Sw maximal component = ',num2str(maxSw),', minimal component = ',num2str(minSw)));

norm_fro = norm(Sw-Sw', 'fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix (Sw) = ',num2str(norm_fro)));
end

norm_fro = norm(Sb-Sb', 'fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix (Sb) = ',num2str(norm_fro)));
end
min_rank = Q-1;
[V,D] = eig(Sb,Sw);

eigenVectors = V;
eigenValues= diag(D);
nr_dimension = size(eigenValues);
for (i=1:nr_dimension)
        if((isreal(eigenValues(i))==1) & (eigenValues(i) > 0.0) & (isinf(eigenValues(i)) == 0)) %real positive eigenvalue
            continue;
        end
        eigenValues(i)=0.0;
end  
[eigenValues,order] = sort(eigenValues, 'descend');
eigenVectors = eigenVectors(:,order);
minnum=min(Q,N);
reduced_dimension = ceil(ratio*minnum);
P = eigenVectors(:, 1:reduced_dimension);
end