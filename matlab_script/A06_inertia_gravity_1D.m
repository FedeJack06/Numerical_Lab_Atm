% Plot dispersion relation for inertia-gravity system
% in different grids and compare with the analytical solution
% -------------------------------------------------------------
% general parameters
% ------------------------------
  f      = 10.^(-4);
  g      = 9.81;
  H      = 1000;
  Ld     = sqrt(g*H)/f;
% ------------------------------
% parameter for example 1
% ------------------------------
  deltax = Ld;
  k      = linspace(0,pi/deltax,100);
  r      = ((2*Ld)/deltax);
  r2     = r.^2;
  sk     = sin(k.*deltax/2);
  ck     = cos(k.*deltax/2);
% ------------------------------
% parameter for example 2
% ------------------------------
  deltax_2 = Ld*4;
  k_2      = linspace(0,pi/deltax_2,100);
  r_2      = ((2*Ld)/deltax_2);
  r2_2     = r_2.^2;
  sk_2     = sin(k_2.*deltax_2/2);
  ck_2     = cos(k_2.*deltax_2/2);
% ------------------------------
% analytical /A/B/C/D grids ex.1
% ------------------------------
  w      = 1 + (g*H.*k.^2)/f^2;
  %wA     = 1 + r2.*sk.^2.*ck.^2;
  wA     = 1 + ((g*H)./(f^2.*deltax.^2)).*sin(k.*deltax).^2;
  wB     = 1 + r2.*sk.^2.;
  wC     = ck.^2 + r2.*sk.^2.;
  wD     = ck.^2 + r2.*sk.^2.*ck.^2;
% ------------------------------
% analytical /A/B/C/D grids ex.2
% ------------------------------
  w2      = 1 + (g*H.*k_2.^2)/f^2;
  w2A     = 1 + r2_2.*sk_2.^2.*ck_2.^2;
  w2B     = 1 + r2_2.*sk_2.^2.;
  w2C     = ck_2.^2 + r2_2.*sk_2.^2.;
  w2D     = ck_2.^2 + r2_2.*sk_2.^2.*ck_2.^2;
% ------------------------------
% Plotting
% ------------------------------
  kp2o= k.*deltax/pi;

  figure('Position',[1 1 900 400]);
   subplot('Position',[.1 .15 .38 .78]);
    plot(kp2o,wA,'g','linewidth',2); hold on
    plot(kp2o,wB,'r','linewidth',2); grid on
    plot(kp2o,wC,'c','linewidth',2);
    plot(kp2o,wD,'m','linewidth',2);
    plot(kp2o,w, 'k','linewidth',2);
    legend('A-grid','B-grid','C-grid','D-grid','Analytical','Location','Northwest');
    ylim([0 12]);set(gca,'fontsize',16);
    title(['$$ R= $$', num2str(r)],'Fontsize',20,'Interpreter','latex');
    xlabel('$$ \frac{k \Delta{x}}{\pi} $$','Interpreter','latex');
    ylabel('$$ \frac{\omega^{2}}{f^{2}} $$','Interpreter','latex');
   subplot('Position',[.6 .15 .38 .78]);
    plot(kp2o,w2A,'g','linewidth',2); hold on
    plot(kp2o,w2B,'r','linewidth',2); grid on
    plot(kp2o,w2C,'c','linewidth',2);
    plot(kp2o,w2D,'m','linewidth',2);
    plot(kp2o,w2, 'k','linewidth',2);
    legend('A-grid','B-grid','C-grid','D-grid','Analytical','Location','Northwest');
    ylim([0 1.8]);set(gca,'fontsize',16);
    title(['$$ R= $$', num2str(r_2)],'Fontsize',20,'Interpreter','latex');
    xlabel('$$ \frac{k \Delta{x}}{\pi} $$','Interpreter','latex');
    ylabel('$$ \frac{\omega^{2}}{f^{2}} $$','Interpreter','latex');
    print -dpng inertia_gravity_1D.png;close
