function [projection_matrix,lower_train_data] = DELIN(train_data,train_p_target, ratio,k,T)
% DELIN [1] deals with partial label dimensionality reduction  by endowing the popular linear discriminant analysis (LDA) techniques with the ability of dealing with partial label training examples.
%
% Syntax
%
%       [projection_matrix,lower_train_data] = DELIN(train_data,train_p_target, ratio,k,T)
%
% Description
%
%       DELIN takes,
%           train_data          - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target      - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           ratio               - the threshold of retained features
%           k                   - the number of the neighboors
%           T                   - the number of iterations
%      
%       and returns,
%           Projection_matrix     - An NxD array, D is the number of retained features,the LDA projection matrix
%           lower_train_data      - An MxD array, a Projected feature space
%          
%
%  [1] J.-H. Wu, M.-L. Zhang. Disambiguation Enabled Linear Discriminant Analysis for Partial Label Dimensionality Reduction (KDD'19), Anchorage, Alaska USA, 2019, in press.
%

M = size(train_data,1); % M--the number of instances
Q = size(train_p_target,1); % Q--the number of labels
Yconfidence = Initialization_labelConfidence(train_p_target);
P = LDA_calculation(train_data,ratio,Yconfidence);
X1=P'*train_data';
X1=X1';

% Find candidate labels for the instance
candidate=cell(M,1);
for i=1:M
    for j=1:Q
        if train_p_target(j,i)==1
            candidate{i}=[candidate{i},j];
        end
    end
end

while T>0
disp(strcat('Current iterations£º ',num2str(T)));
temp=Yconfidence;
EstimatedY=cell(1,M);
Neighbor_votes=zeros(1,M);
[EstimatedY,Neighbor_votes]=KNN_disambiguation(X1,k,Q,candidate,Yconfidence);
        
% update label confidence
for i=1:M
    sizeguess=size(EstimatedY{i},2); 
    guesslabel=EstimatedY{1,i};
    if size(guesslabel)==size(candidate{i})
        if guesslabel==candidate{i}
        continue;
        end
    else
        diffe=setdiff(candidate{i},guesslabel);
        sizediffe=size(diffe,2);
        for j=1:sizeguess
            Yconfidence(i,guesslabel(j))=(Neighbor_votes(i))/k; 
        end
        for p=1:sizediffe
            Yconfidence(i,diffe(p))=(1-(Neighbor_votes(i)*sizeguess)/k)/sizediffe;
        end
    end
end
 for i=1:M
            if(sum(Yconfidence(i,:))<=0) 
                continue;
            end
            Yconfidence(i,:) = Yconfidence(i,:)/norm(Yconfidence(i,:),1);
end

if temp==Yconfidence
    break;
else
    P = LDA_calculation(train_data,ratio,Yconfidence);
    X1=P'*train_data';
    X1=X1';
end
T=T-1;
end
projection_matrix=P;
lower_train_data=X1;
end