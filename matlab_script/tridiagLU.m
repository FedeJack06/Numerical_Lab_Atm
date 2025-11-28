function [e,f] = tridiagLU(a,b,c)
% tridiagLU Obtain the LU factorization of a tridiagonal matrix
%
% Synopsis: [e,f] = tridiag(a,b,c)
%
% Input: a,b,c = vectors defining the tridiagonal matrix. a is the
% subdiagonal, b is the main diagonal, and c is the superdiagonal
%
% Output: e,f = vectors defining the L and U factors of the tridiagonal matrix
n = length(a);
e = zeros(n,1); f = e;
e(1) = b(1);
f(1) = c(1)/b(1);
for i=2:n
e(i) = b(i) - a(i)*f(i-1);
f(i) = c(i)/e(i);
end
