% -------------------------------------
% Four:
% Analyze Truncation error
% Fix delta-t and change delta-x
% compare the results with the true solution
% remember stablity of FTCS depends both on delta-x and delta-t (and K)
% -------------------------------------

 nt    =  1001; % Number of time-steps
 nx    =  20; % Number of mesh points
 alpha = 0.1; % Diff coeff.
 L     =   1.1; % Spatial Length of the domain
 tmax  = 1.0; % Temporal integration length

% -------------------------------------
% Compute exact solution
% -------------------------------------

 nx=[76:-4:4];
 dx = L./(nx-1);
 dt = tmax/(nt-1);
  r = (alpha*dt)./(dx.^2);

 for tt=1:size(nx,2)
   [err_ftcs(tt),x,t,U_ftcs] = heat_eq_FTCS(nt,nx(tt),alpha,L,tmax,0);
   [err_btcs(tt),x,t,U_btcs] = heat_eq_BTCS(nt,nx(tt),alpha,L,tmax,0);
   [err_cn(tt)  ,x,t,U_cn]   =   heat_eq_CN(nt,nx(tt),alpha,L,tmax,0);
 end
 
figure('Position',[1 1 600 500]);
 plot(dx(1:end),err_ftcs(1:end),'--r','linewidth',2); hold on;
 plot(dx(1:end),err_btcs(1:end),'--b','linewidth',2); grid on;
 plot(dx(1:end),  err_cn(1:end),'--g','linewidth',2); 
 title(['$$ \Delta{t} = $$' num2str(dt)],'Fontsize',16,'interpreter','latex');
 set(gca,'Fontsize',18);% axis equal
 xlabel('$$ \Delta{x} $$','interpreter','latex');
 ylabel('err','interpreter','latex');
 print -dpng heat_fig03.png;close;
