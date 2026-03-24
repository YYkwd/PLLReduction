clc;
clear;
close;

load('sample_data.mat');

T = 50;
mu = 0.5;
dim_para = 0.999;
k = 8;

[projection_matrix, lower_train_data] = CENDA(data, partial_target, T, mu, dim_para, k);
