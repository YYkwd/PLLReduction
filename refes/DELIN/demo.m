clc;
clear;
close;

load('sample_data.mat');

ratio = 0.6;
k = 8;
T = 75;
[projection_matrix, lower_train_data] = DELIN(data, partial_target, ratio, k, T);

ratio = 0.6;
k = 8;
T = 15;

train_data = data';
train_p_target = partial_target';

[Theta, lower_train_data] = KDELIN(train_data, train_p_target, ratio, k, T);
