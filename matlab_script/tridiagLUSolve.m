function v = tridiagLUsolve(d,a,e,f,v)
% tridiagLUsolve Solve (LU)*v = d where L and U are LU factors of a tridiagonal
% matrix
%
% Synopsis: v = tridiagLUsolve(d,e,f)
% v = tridiagLUsolve(d,e,f,v)
%
% Input: d = right hand side vector of the system of equatoins
% e,f = vectors defining the L and U factors of the tridiagonal matrix.
% e and f are obtained with the tridiagLU function
% v = solution vector. If v is supplied, the elements of v are over-
% written (thereby saving the memory allocation step). If v is not
% supplied, it is created. v is used as a scratch vector in the
% forward solve.
%
% Output: v = solution vector
n = length(d);
if nargin<5, v = zeros(n,1); end
% --- Forward substitution to solve L*w = d
v(1) = d(1)/e(1);
for i=2:n
v(i) = (d(i) - a(i)*v(i-1))/e(i);
end
% --- Backward substitution to solve U*v = w
for i=n-1:-1:1
v(i) = v(i) - f(i)*v(i+1);
end
