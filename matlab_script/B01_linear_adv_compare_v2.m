% ---------------------------------------------------------
% Compare different Linear 1D Advection numerical methods
%  One-sided Upwind
%  Lax-Wendroff
%  TVD (with different limiters)
% ---------------------------------------------------------

% ---------------------------------------------------------
% Parameters
% ---------------------------------------------------------
    a     = 0.5; % Advective speed [m/s]
    dx    = 0.01; % [m]
    dt    = 0.005; % [s]
    cfl   = a*dt/dx;
% ---------------------------------------------------------
% Domain Discretization
% ---------------------------------------------------------
    x  = 0:dx:1;
% ---------------------------------------------------------
% Initial Condition
% ---------------------------------------------------------
    n                 = length(x);
    u_0               = ones(1,n)*0.1;
    u_0 = [((x>=0.1) & (x <= 0.3)) | ((x >= 0.5) & (x <= 0.7))]*1;
    u_next = zeros(1,n);

% ---------------------------------------------------------
% Experiment with different duration to show the evolution
% of the solution
% ---------------------------------------------------------
    tend1  = 1.0; % End time
    t  = 0:dt:tend1;
    shift    = ceil((a/dx)*(tend1+dt))
    u_now1=circshift(u_0,shift);
% ---------------------------------------------------------
% TVD
% ---------------------------------------------------------
    [u_TVD11]=B02_TVD(dt,dx,x,t,u_0,a,1);
    [u_TVD21]=B02_TVD(dt,dx,x,t,u_0,a,2);
    [u_TVD31]=B02_TVD(dt,dx,x,t,u_0,a,3);
% ---------------------------------------------------------
% Upwind
% ---------------------------------------------------------
    [u_UPW1]=B02_UpWind(dt,dx,x,t,u_0,a);
% ---------------------------------------------------------
% Lax-Wendroff
% ---------------------------------------------------------
    [u_LW1]=B02_LW(dt,dx,x,t,u_0,a);
% ---------------------------------------------------------
    tend2  = 4.0; % End time
    t  = 0:dt:tend2;
    shift    = ceil((a/dx)*(tend2+dt))
    u_now2=circshift(u_0,shift);
% ---------------------------------------------------------
% TVD
% ---------------------------------------------------------
    [u_TVD12]=B02_TVD(dt,dx,x,t,u_0,a,1);
    [u_TVD22]=B02_TVD(dt,dx,x,t,u_0,a,2);
    [u_TVD32]=B02_TVD(dt,dx,x,t,u_0,a,3);
% ---------------------------------------------------------
% Upwind
% ---------------------------------------------------------
    [u_UPW2]=B02_UpWind(dt,dx,x,t,u_0,a);
% ---------------------------------------------------------
% Lax-Wendroff
% ---------------------------------------------------------
    [u_LW2]=B02_LW(dt,dx,x,t,u_0,a);

% ---------------------------------------------------------
% Plot
% ---------------------------------------------------------

   figure('Position',[10 10 900 600]);
     subplot('Position',[0.1 0.55 0.83 0.35]);
      plot(x,u_now1,'-k','Linewidth',2); hold on
      plot(x,u_UPW1,'-.r','Linewidth',1.5);
      plot(x,u_LW1    ,':b' ,'Linewidth',1.5); 
      plot(x,u_TVD11  ,'-g' ,'Linewidth',2); 
      plot(x,u_TVD21,'--m','Linewidth',2); 
      plot(x,u_TVD31,'-y','Linewidth',2); 
      grid minor; box on;
      ylabel('[-]','interpreter','latex','fontsize',16);
      ylim([-0.5,1.5]);
      title(['$$ \Delta x= $$' num2str(dx) ' $$ \Delta t = $$' num2str(dt) ' CFL=' num2str(cfl) ' End-Time=' num2str(tend1)],'Interpreter','latex','Fontsize',20);
      set(gca,'Fontsize',18,'Xticklabel','');
     subplot('Position',[0.1 0.1 0.83 0.35]);
      plot(x,u_now2,'-k','Linewidth',2); hold on
      plot(x,u_UPW2,'-.r','Linewidth',1.5);
      plot(x,u_LW2    ,':b' ,'Linewidth',1.5); 
      plot(x,u_TVD12  ,'-g' ,'Linewidth',2); 
      plot(x,u_TVD22,'--m','Linewidth',2); 
      plot(x,u_TVD32,'-y','Linewidth',2); 
      legend('Analytical','UPW','LW','Van Leer','Superbeed','MinMod','Interpreter','late');
      grid minor; box on;
      xlabel('X','interpreter','latex','fontsize',16);
      ylabel('[-]','interpreter','latex','fontsize',16);
      ylim([-0.5,1.5]);
      title(['$$ \Delta x= $$' num2str(dx) ' $$ \Delta t = $$' num2str(dt) ' CFL=' num2str(cfl) ' End-Time=' num2str(tend2)],'Interpreter','latex','Fontsize',20);
      set(gca,'Fontsize',18);

  print -dpng linear_adv_schemes.png;close;
