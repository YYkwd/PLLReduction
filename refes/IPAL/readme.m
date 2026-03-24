%This is an examplar file on how the IPAL program could be used (The main function is "IPAL_train.m" and "IPAL_predict.m")
%
%Type 'help IPAL_train' and 'help IPAL_predict' under Matlab prompt for more detailed information
%

load('sample data.mat'); % Loading the file containing the necessary inputs for calling the IPAL function

k = 10;                  %set the number of nearest neighbors
alpha = 0.95;            %set the balancing coefficient

model = IPAL_train(train_data,train_p_target,k,alpha);                     %disambiguation phase 
[accuracy,predictLael] = IPAL_predict(model,test_data,test_target,k);      %testing phase
fprintf('classification accuracy: %f\n',accuracy);