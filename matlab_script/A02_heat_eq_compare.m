% -------------------------------------
% third:
% Analyze Truncation error
% Fix delta-x and change delta-T
% compare the results with the true solution
% remember stablity of FTCS depends both on delta-x and delta-t (and K)
% -------------------------------------
close all;clear all;

 nx    = 201; % Number of mesh points
 alpha = 0.1; % Diff coeff.
 L     =   1; % Spatial Length of the domain
 tmax  = 0.25; % Temporal integration length

% -------------------------------------
% Compute exact solution
% -------------------------------------

 nt=[10000:-500:2000 1999:-1:1980];
 dx = L/(nx-1);
 dt = tmax./(nt-1);
  r = (alpha.*dt)./(dx.^2);

 for tt=1:size(nt,2)
   [err_ftcs(tt),x,t,U_ftcs] = heat_eq_FTCS(nt(tt),nx,alpha,L,tmax,0);
   [err_btcs(tt),x,t,U_btcs] = heat_eq_BTCS(nt(tt),nx,alpha,L,tmax,0);
   [err_cn(tt)  ,x,t,U_cn]   =   heat_eq_CN(nt(tt),nx,alpha,L,tmax,0);
   clear U_*
 end

  figure('Position',[1 1 600 900]);
  a1=axes('Position',[.15 .75 .8 .2]);
   plot(dt,err_ftcs,'--r','linewidth',2); hold on;
   plot(dt,err_btcs,'--b','linewidth',2); grid on;
   plot(dt,  err_cn,'--g','linewidth',2); 
   title(['$$ A) \Delta{x} = $$' num2str(dx)],'interpreter','latex','Fontsize',16);
   xlabel('$$ \Delta{t} $$','interpreter','latex'); ylabel('err','interpreter','latex');
   set(gca,'Fontsize',18); ylim([0 0.0005]);
  a2=axes('Position',[.15 .4 .8 .2]);
   plot(r,err_ftcs,'--r','linewidth',2); hold on;
   plot(r,err_btcs,'--b','linewidth',2); grid on;
   plot(r,  err_cn,'--g','linewidth',2); 
   title(['$$ B) \Delta{x} = $$' num2str(dx)],'interpreter','latex','Fontsize',16);
   xlabel('$$ r $$','interpreter','latex');
   ylabel('err','interpreter','latex');
   set(gca,'Fontsize',18); ylim([0 0.0005]);xlim([.075 .555]);
drawnow
%print -dpng heat_fig04.png;close;

clear err*
 nt=[2000:-1:1];
 dx = L/(nx-1);
 dt = tmax./(nt-1);
 for tt=1:size(nt,2)
   [err_btcs(tt),x,t,U_btcs] = heat_eq_BTCS(nt(tt),nx,alpha,L,tmax,0);
   [err_cn(tt)  ,x,t,U_cn]   =   heat_eq_CN(nt(tt),nx,alpha,L,tmax,0);
   clear U_*
 end
  a3=axes('Position',[.15 .1 .8 .2]);
   plot(dt,  err_cn,'--g','linewidth',2); hold on             
   plot(dt,err_btcs,'--b','linewidth',2); grid on;
   title(['$$ C) \Delta{x} =  $$' num2str(dx)],'interpreter','latex','Fontsize',16);
   xlabel('$$ \Delta{t} $$','interpreter','latex'); ylabel('err');
   ylabel('err','interpreter','latex');
   set(gca,'Fontsize',18); ylim([0 0.01]);

 print -dpng heat_fig02.png; close;
