function [errout,xo,to,Uo] = heat_eq_FTCS(nt,nx,alpha,L,tmax,errPlots)
%
% -----------------------------------------------------------------
% Solve 1D heat equation with the BTCS scheme
% -----------------------------------------------------------------
%
% Calling: 
% ---------
% heat_eq_FTCS(nt)
% heat_eq_FTCS(nt,nx)
% heat_eq_FTCS(nt,nx,alpha)
% heat_eq_FTCS(nt,nx,alpha,L)
% heat_eq_FTCS(nt,nx,alpha,L,tmax)
% heat_eq_FTCS(nt,nx,alpha,L,tmax,errPlots)
% err = heatFTCS(...)
% [err,x,t,U] = heatFTCS(...)
%
% Input:
% ---------
% nt       = number of steps. Default: nt = 10;
% nx       = number of mesh points in x direction. Default: nx=20
% alpha    = diffusivity. Default: alpha = 0.1
% L        = length of the domain. Default: L = 1;
% tmax     = maximum time for the simulation. Default: tmax = 0.5
% errPlots = flag (1 or 0) to control whether plots of the solution
%            and the error at the last time step are created.
%            Default: errPlots = 1 (make the plots)
%
% Output: 
% ---------
% err = L2 norm of error evaluated at the spatial nodes on last time step
% x   = location of finite difference nodes
% t   = values of time at which solution is obtained (time nodes)
% U   = matrix of solutions: U(:,j) is U(x) at t = t(j)
% ----------------------------------------------------------------

    if nargin<1, nt = 10; end
    if nargin<2, nx = 20; end
    if nargin<3, alpha = 0.1; end
    if nargin<4, L = 1; end
    if nargin<5, tmax = 0.5; end
    if nargin<6, errPlots=1; end

% --- Compute mesh spacing and time step

    dx = L/(nx-1);
    dt = tmax/(nt-1);
    r = alpha*dt/dx^2; r2 = 1 - 2*r;

% --- Create arrays to save data for export

    x = linspace(0,L,nx)';
    t = linspace(0,tmax,nt);
    U = zeros(nx,nt);

% --- Set IC and BC

    U(:,1) = sin(pi*x/L); % implies u0 = 0; uL = 0;
    u0 = 0; uL = 0; % needed to apply BC inside time step loop

% --- Loop over time steps

    for m=2:nt
        for i=2:nx-1
            U(i,m) = r*U(i-1,m-1) + r2*U(i,m-1) + r*U(i+1,m-1);
        end
    end

% --- Compare with exact solution at end of simulation

    ue = sin(pi*x/L)*exp(-t(nt)*alpha*(pi/L)^2);
    err = norm(U(:,nt)-ue);

% --- Set values of optional output variables

    if nargout>0, errout = err; end
    if nargout>1, xo = x; to = t; Uo = U; end

    fprintf('\tdt, dx, r , err= %12.3e %12.3e %8.3f %12.3e\n',dt,dx,r,err);
% --- Plot error in solution at the last time step

    if ~errPlots, return; end
    fprintf('\nNorm of error = %12.3e at t = %8.3f\n',err,t(nt));
    fprintf('\tdt, dx, r = %12.3e %12.3e %8.3f\n',dt,dx,r);
    figure; plot(x,U(:,nt),'o--',x,ue,'-'); xlabel('x'); ylabel('u');
    title(['dt=' num2str(dt)]);
    legend('FTCS','Exact');
%    figure; plot(x,U(:,nt)-ue,'o--'); xlabel('x'); ylabel('u - u_e');
