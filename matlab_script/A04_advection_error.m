%
  deltax = 0.1;
  c      = 2.0;

  k      = [0:1:32];
% real solution
  w      = c.*k;
  wpr    = w.*(deltax/c);
% Second order scheme
  w      = (c/deltax).*sin(k*deltax);
  wp2o   = w.*(deltax/c);
%fourth order scheme
  w      = (c/(6*deltax)).*(-sin(2*k.*deltax)+8*sin(k.*deltax));
  wp4o   = w.*(deltax/c);

  kp2o= k.*deltax/pi;
 figure;
  plot(kp2o,wp2o,'r','linewidth',2); hold on
  plot(kp2o,wp4o,'g','linewidth',2); hold on
  plot(kp2o,wpr,'k','linewidth',2); grid on
  xlim([0 1]);
  set(gca,'fontsize');
  legend('$$ 2^{nd} Order $$', ...
         '$$ 4^{th} Order $$','analytical','interpreter','latex');
  xlabel('$$ \frac{k \Delta{x}}{\pi} $$','Interpreter','latex');
  ylabel('$$ \frac{\omega \Delta{x}}{c} $$','Interpreter','latex');
print -dpng adv_pha_err.png;close;
