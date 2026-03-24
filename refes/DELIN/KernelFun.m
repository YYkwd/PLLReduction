function [k] = KernelFun(a, b)
%   Function KERNELFUN 
% 
%   set kernel function
% 
%   Linear
% k = a' * b; 
% 
%   Polynomial
% c1 = 0;
% c2 = 0;
% c3 = 0;
% k = (c1 * a' * b + c2)^c3;

%   Gaussian
sigma = 0.5;
k = exp(-sum((a - b).^2) / (2 * sigma * sigma));

%   Sigmoid
% c1 = 0;
% c2 = 0;
% k = tanh(c1 * a' * b + c2);
% 
end
