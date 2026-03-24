function [Yconfidence] = Initialization_labelConfidence(train_p_target)
% The function Initialization_labelConfidence initializes the label confidence matrix.
%
% Syntax
%
%       [Yconfidence] = Initialization_labelConfidence(train_p_target)
%
% Description
%
%       Initialization_labelConfidence takes,
%           train_p_target      - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%          
%       and returns,
%           Yconfidence         - An MxQ array, Yconfidence(i,j) denotes the confidence of the jth class label as its true label in the ith instance
%

[Q,M]=size(train_p_target);
Yconfidence=zeros(M,Q);
for i=1:M
    sum1=0;
    for j=1:Q
        if  train_p_target(j,i)==1
            sum1=sum1+1;
        end   
    end
    for j=1:Q
        if  train_p_target(j,i)==1
            Yconfidence(i,j)=1/sum1;
        end
    end
   
end
 for i=1:M
            if(sum(Yconfidence(i,:))<=0) 
                continue;
            end
            Yconfidence(i,:) = Yconfidence(i,:)/norm(Yconfidence(i,:),1);
end
end