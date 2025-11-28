% ----------------------------------------------------------
% compare results of different schemes to solve the heat equation
% Forward in Time Centred in Space (FTCS)
% Backward in Time Centred in Space (BTCS)
% Trapeziodal scheme (CN)
% ----------------------------------------------------------

% -------------------------------------
% First with default parameters
% -------------------------------------
 nt    =  10; % Number of time-steps
 nx    =  20; % Number of mesh points
 alpha = 0.1; % Diff coeff.
 L     =   1; % Spatial Length of the domain
 tmax  = 0.5; % Temporal integration length

% -------------------------------------
% Call FTCS BTCS and CN
% -------------------------------------
 [err_ftcs,x,t,U_ftcs] = heat_eq_FTCS(nt,nx,alpha,L,tmax,0);
 [err_btcs,x,t,U_btcs] = heat_eq_BTCS(nt,nx,alpha,L,tmax,0);
 [err_cn  ,x,t,U_cn  ] = heat_eq_CN(  nt,nx,alpha,L,tmax,0);

% -------------------------------------
% Compute exact analytical solution
% -------------------------------------
  ue = sin(pi*x/L)*exp(-t(nt)*alpha*(pi/L)^2);

% -------------------------------------
% Plotting
% -------------------------------------
 figure('Position',[1 1 1000 400]);
   a1=axes('Position',[0.055 0.15 .29 .75]);
   plot(x,    ue      ,'k'  ,'linewidth',2); hold on
   plot(x,U_ftcs(:,nt),'--r','linewidth',2); 
   plot(x,U_btcs(:,nt),'--b','linewidth',2); grid on
   plot(x,  U_cn(:,nt),'--g','linewidth',2);
   title('A)','Interpreter','latex');
   set(gca,'fontsize',18);
   xlabel('X','Interpreter','latex');ylabel('$$ \phi $$','Interpreter','latex');
   legend('Analytical','FTCS','BTCS','CN','Interpreter','latex');
   set(gca,'xtick',[0:.2:1],'ytick',[0:.2:1]);
   ylim([0 1]);

% -------------------------------------
% Second:
% Show instabilities in FTCS
% -------------------------------------
 nt    =  20; % Number of time-steps
 tmax  = 1.0; % Temporal integration length
 [err_ftcs,x,t,U_ftcs] = heat_eq_FTCS(nt,nx,alpha,L,tmax,0);
 [err_btcs,x,t,U_btcs] = heat_eq_BTCS(nt,nx,alpha,L,tmax,0);
 [err_cn  ,x,t,U_cn  ] = heat_eq_CN(  nt,nx,alpha,L,tmax,0);

% -------------------------------------
% Plotting
% -------------------------------------
   a2=axes('Position',[.375 0.15 .29 .75]);
   plot(x,U_ftcs,'--r','linewidth',1); grid on;
   title('B) FTCS','Interpreter','latex');
   set(gca,'xtick',[0:.2:1],'ytick',[0:.2:1],'yticklabel','');
   xlabel('X','Interpreter','latex');
   set(gca,'fontsize',18);

   a2=axes('Position',[0.7 0.15 .29 .75]);
   plot(x,U_cn  ,'--g','linewidth',1); hold on;
   plot(x,U_btcs,'--b','linewidth',1); grid on;
   title('C) BTCS and CN','interpreter','latex');
   set(gca,'xtick',[0:.2:1],'ytick',[0:.2:1],'yticklabel','');
   xlabel('X','Interpreter','latex');
   set(gca,'fontsize',18);

% -------------------------------------
 print -dpng heat_fig01.png; close;
