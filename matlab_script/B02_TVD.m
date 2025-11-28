function [u] = TVD(dt,dx,x,t,u_0,a,limiter)

% ---------------------------------------------------------
% Parameters
% ---------------------------------------------------------

   a_m = min(0,a);
   a_p = max(0,a);
   n = length(x);

% ---------------------------------------------------------
% Initilize vector variables
% ---------------------------------------------------------
   r       = zeros(1,n);
   F_rl    = zeros(1,n);
   F_rh    = zeros(1,n);
   F_ll    = zeros(1,n);
   F_lh    = zeros(1,n);
   F_right = zeros(1,n);
   F_left  = zeros(1,n);
   u_next  = zeros(1,n);

% ---------------------------------------------------------
% Set Initial condition
% ---------------------------------------------------------
    u = u_0;

% ---------------------------------------------------------
% Start temporal Loop
% ---------------------------------------------------------
for k = t

        % BC
        u_next(1) = u_next(end-1);
        u_next(n) = u_next(2);

    for j = 2:n-1
        % smooth measurement factor 'r'
        if u(j) == u(j+1)
            r(j) = 1;
        elseif a > 0
            r(j) = (u(j) - u(j-1)) / (u(j+1) - u(j));
        elseif a < 0
            r(j) = (u(j+2) - u(j+1)) / (u(j+1) - u(j));
        end
        r(1) = 1; r(n) = 1;
    end
        
        % -------------------------------------------------
        % Define Flux Limiter function:
        % -------------------------------------------------
        % (Van Leer 1974)
        % -------------------- 
          if (limiter==1)
          phi = (r + abs(r))./(1 + abs(r));
        end

        % (Superbeed, Roe 1985)
        % -------------------- 
        if (limiter==2)
          phi1 = min(2,r);
          phi2 = min(1,2.*r);
          phi  = max(max(0,phi1),min(2,r));
        end
        
        % (minmod)
        % -------------------- 
        if (limiter==3)
           phi = max(0,min(1,r));
        end

    for j = 2:n-1    
        % Compute fluxes 
        F_rl(j) = a_p*u(j) + a_m*u(j+1);
        F_rh(j) = (1/2)*a*(u(j)+u(j+1)) - (1/2)*(a^2)*(dt/dx)*(u(j+1)-u(j));
        
        F_ll(j) = a_p*u(j-1) + a_m*u(j);
        F_lh(j) = (1/2)*a*(u(j-1)+u(j)) - (1/2)*(a^2)*(dt/dx)*(u(j)-u(j-1));
        
        % Advance in time
        F_right(j) = F_rl(j) + phi(j)*( F_rh(j) - F_rl(j) );
        F_left(j)  = F_ll(j) + phi(j-1)*( F_lh(j) - F_ll(j) );
        
        u_next(j) = u(j) - dt/dx*(F_right(j) - F_left(j));
    end
    
        % Swap array 
        u = u_next(1:n);
end
    
end
