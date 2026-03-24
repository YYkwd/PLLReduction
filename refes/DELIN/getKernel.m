function [K] = getKernel(A, B)
%   Function GETKERNEL 

r = size(A, 2);     %   number of rows of K
c = size(B, 2);     %   number of cols of K

K = zeros(r, c);

for i = 1:r
    for j = 1:c
        K(i, j) = KernelFun(A(:, i), B(:, j));
    end
end

end
