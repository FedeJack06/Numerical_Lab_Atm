function [u] = upwind(dt,dx,x,t,u_0,a)
    u_upwind = u_0;
    cfl=a*dt/dx;
    n                 = length(x);
    u_next = zeros(1,n);

for k = t
    % BC
    u_upwind(1) = u_upwind(end-1);
    u_upwind(n) = u_upwind(2);

    for j = 2:n-1
    u_next(j) = u_upwind(j) - cfl*(u_upwind(j)-u_upwind(j-1));
    end
    u_upwind(2:end-1) = u_next(2:end-1);
end
    u = u_upwind;


end
