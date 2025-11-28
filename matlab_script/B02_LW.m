function [u] = LW(dt,dx,x,t,u_0,a)
    u_LW = u_0;
    cfl=a*dt/dx;
    n                 = length(x);
    u_next = zeros(1,n);
for k = t
    % BC
    u_LW(1) = u_next(end-2);
    u_LW(n) = u_next(3);
    for j = 2:n-1
    u_next(j) = u_LW(j) - 1/2*cfl*(u_LW(j+1)-u_LW(j-1)) + ...
        1/2*(cfl^2)*(u_LW(j+1)-2*u_LW(j)+u_LW(j-1));
    end
    u_LW = u_next;
end


    u = u_LW;


end
