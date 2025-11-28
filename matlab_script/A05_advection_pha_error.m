%
  deltax = 0.1;
  k      = [0:1:32];
  g      = 9.81;
  H      = 1000;
  gH     = g*H;
  sqrt(gH)
% real solution
  w      = k.*sqrt(gH);
  wpr    = (w.*deltax)/sqrt(gH);
% Second order scheme
  w      = sqrt(-(gH/(4*(deltax.^2))).*(2*cos(2*k*deltax) - 2));
  wp2o   = (w.*deltax)/sqrt(gH);
% Second order staggered 
  w      = sqrt(-(gH/((deltax.^2))).*(2*cos(k*deltax) - 2));
  wp4o   = (w.*deltax)/sqrt(gH);

  kp2o= k.*deltax/pi;

 figure;
  plot(kp2o,wp2o,'r','linewidth',2); hold on
  plot(kp2o,wp4o,'g','linewidth',2); hold on
  plot(kp2o,wpr,'k','linewidth',2); grid on
  xlim([0 1]);
  set(gca,'fontsize',16);
  legend('$$ 2^{nd} Order $$', ...
         '$$ 2^{th} Order :\ staggered $$','analytical','interpreter','latex');
  xlabel('$$ \frac{k \Delta{x}}{\pi} $$','Interpreter','latex');
  ylabel('$$ \frac{\omega \Delta{x}}{c_{g}} $$','Interpreter','latex');

print -dpng grav_pha_err.png;close;
