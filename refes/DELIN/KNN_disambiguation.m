function [EstimatedY,Neighbor_votes] = KNN_disambiguation(lower_train_data,k,Q,candidate,Yconfidence)
% 
% KNN_disambiguation : Disambiguation using k-nearest neighbor information.
%
% Syntax
%
%       [EstimatedY,Neighbor_votes] = KNN_disambiguation(lower_train_data,k,Q,candidate,Yconfidence)
%
% Description
%
%       LDA_calculation takes,
%           lower_train_data    - An MxD array, a Projected feature space
%           k                   - the number of the neighboors
%           Q                   - the number of labels
%           candidate           - candidate label sets
%           Yconfidence         - An MxQ array, Yconfidence(i,j) denotes the confidence of the jth class label as its true label in the ith instance         
%      
%       and returns,
%           EstimatedY          - Predictive label obtained after disambiguation
%           Neighbor_votes      - Number of votes for predictive label by k-nearest neighbors
%
M=size(lower_train_data,1);
kdtree = KDTreeSearcher(lower_train_data); 
[neighbor,dis] = knnsearch(kdtree,lower_train_data,'k',k+1); 
neighbor = neighbor(:,2:end);
wr=zeros(1,k);
for i=1:k
    wr(i)=k-i+1;          
end
for i=1:M
    sumY=zeros(1,Q);       
    count1=zeros(1,Q);        
    sizecandidate=size(candidate{i},2); 
    percandidate= candidate{i,1};    
    for t=1:sizecandidate
        for j=1:k
            indexneighbor=neighbor(i,j); 
            indexlabel=percandidate(t); 
            sumY(indexlabel)=sumY(indexlabel)+Yconfidence(indexneighbor,indexlabel)*wr(j);
            if Yconfidence(indexneighbor,indexlabel)>0
                count1(indexlabel)=count1(indexlabel)+1; 
            end
        end
    end
   
    [row,col]=find(sumY==max(sumY)); 
    count2=size(col,2); 
    count3=count1(col);  
    count4=max(count3); 
    [~,count5]=find(count1==count4);
    if (max(sumY)==0)     
        EstimatedY{i}=candidate{i};
        Neighbor_votes(i)=0;
        
    else if count2==1
        EstimatedY{i}=col;
        Neighbor_votes(i)=count3;
       
    else if (size(count4,2)==1)
      
         Neighbor_votes(i)=count4;
         EstimatedY{i}=count5;
    else
         EstimatedY{i}=count5;
         Neighbor_votes(i)=count4;
          
        end
        end
    end
     numgess=size(EstimatedY{i},2);
     if numgess>1
         numk=Neighbor_votes(i)*numgess;
         if numk>k
             ran=randperm(numgess);
             EstimatedY{i}=EstimatedY{i}(ran(1));
         end
     end
         
end
end