% Plot dispersion relation for inertia-gravity system
% in different grids and compare with the analytical solution
% 2D case
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
  deltax = Ld*0.5;
  deltay = deltax;
  k      = linspace(0,pi/deltax,100);
  l      = linspace(0,pi/deltay,100);
  r      = ((Ld)/deltax); % larger the r better the waves are resolved
% ------------------------------
% parameter for example 2
% ------------------------------
  deltax_2 = Ld*4;
  deltay_2 = deltax_2;
  k_2      = linspace(0,pi/deltax_2,100);
  l_2      = linspace(0,pi/deltay_2,100);
  r_2      = ((Ld)/deltax_2);
% ------------------------------
% Plotting parameters
% ------------------------------
  min_max1=[0 7.5];
  min_max2=[.5 1.5];
% ------------------------------
% Relation dispersions ex.1
% ------------------------------
  for jpi = 1 : size(k,2);
   for jpj = 1 : size(l,2);
% Analytical
       w(jpi,jpj) = sqrt(f^2+(g*H)*(k(jpi)^2+l(jpj)^2));
% A-grid
       wA(jpi,jpj) = sqrt( ...
                          f^2+((g*H)/(deltax^2)) * ... 
                    (sin(k(jpi)*deltax)^2+sin(l(jpj)*deltax)^2) ...
                         );
% B-grid
       wB(jpi,jpj) = sqrt( ...
                          f^2+(4*(g*H)/(deltax^2)) * ... 
                    ( ...
                     sin(k(jpi)*deltax/2)^2*cos(l(jpj)*deltax/2)^2 + ...
                     cos(k(jpi)*deltax/2)^2*sin(l(jpj)*deltax/2)^2 ...
                    ));
% C-grid
       wC(jpi,jpj) = sqrt( ...
                          f^2*cos(k(jpi)*deltax/2)^2*cos(l(jpj)*deltax/2)^2 + ...
                          4*((g*H)/(deltax^2)) * ...
                          ( sin(k(jpi)*deltax/2)^2 + sin(l(jpj)*deltax/2)^2) ...
                         );
   end
  end

% ------------------------------
% analytical /A/B/C/D grids ex.2
% ------------------------------
  for jpi = 1 : size(k_2,2);
   for jpj = 1 : size(l_2,2);
% Analytical
       w2(jpi,jpj) = sqrt(f^2+(g*H)*(k_2(jpi)^2+l_2(jpj)^2));
% A-grid
       w2A(jpi,jpj) = sqrt( ...
                          f^2+((g*H)/(deltax_2^2)) * ... 
                    (sin(k_2(jpi)*deltax_2)^2+sin(l_2(jpj)*deltax_2)^2) ...
                         );
% B-grid
       w2B(jpi,jpj) = sqrt( ...
                          f^2+(4*(g*H)/(deltax_2^2)) * ... 
                    ( ...
                     sin(k_2(jpi)*deltax_2/2)^2*cos(l_2(jpj)*deltax_2/2)^2 + ...
                     cos(k_2(jpi)*deltax_2/2)^2*sin(l_2(jpj)*deltax_2/2)^2 ...
                    ));
% C-grid
       w2C(jpi,jpj) = sqrt( ...
                          f^2*cos(k_2(jpi)*deltax_2/2)^2*cos(l_2(jpj)*deltax_2/2)^2 + ...
                          4*((g*H)/(deltax_2^2)) * ...
                          ( sin(k_2(jpi)*deltax_2/2)^2 + sin(l_2(jpj)*deltax_2/2)^2) ...
                         );
   end
  end
% ------------------------------
% Plotting
% ------------------------------
  load myc
  kp2o= k.*deltax/pi;
  lp2o= l.*deltax/pi;

  kp22o= k_2.*deltax_2/pi;
  lp22o= l_2.*deltax_2/pi;
  figure('Position',[1 1 600 900]);
   a1=axes('Position',[.1 .78 .4 .19]);
       pcolor(kp2o,lp2o, w/f); shading flat;hold on
       [cc,hh]=contour(kp2o,lp2o, w/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       title(['r=' num2str(r)],'Interpreter','latex','Fontsize',20);caxis(min_max1);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'xticklabel','','ytick',[0:.2:1]);
       ylabel('l','Interpreter','latex');
       text(1.05,.65,'Analytic','rotation',-90,'Interpreter','latex','fontsize',20);
   a2=axes('Position',[.1 .56 .4 .19]);
       pcolor(kp2o,lp2o,wA/f); shading flat; hold on;
       [cc,hh]=contour(kp2o,lp2o,wA/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);caxis(min_max1);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'xticklabel','','ytick',[0:.2:1]);
       ylabel('l','Interpreter','latex');
       text(1.05,.65,'A-Grid','rotation',-90,'Interpreter','latex','fontsize',20);
   a3=axes('Position',[.1 .34 .4 .19]);
       pcolor(kp2o,lp2o,wB/f); shading flat;hold on; caxis(min_max1);
       [cc,hh]=contour(kp2o,lp2o,wB/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'xticklabel','','ytick',[0:.2:1]);
       ylabel('l','Interpreter','latex');
       text(1.05,.65,'B-Grid','rotation',-90,'Interpreter','latex','fontsize',20);
   a4=axes('Position',[.1 .12 .4 .19]);
       pcolor(kp2o,lp2o,wC/f); shading flat;hold on;caxis(min_max1);
       [cc,hh]=contour(kp2o,lp2o,wC/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'ytick',[0:.2:1]);
       xlabel('k','Interpreter','latex');
       ylabel('l','Interpreter','latex');
       text(1.05,.65,'C-Grid','rotation',-90,'Interpreter','latex','fontsize',20);
       colorbar(a4,'south','Position',[.1 .03 .4 .02]);

   a5=axes('Position',[.55 .78 .4 .19]);
       pcolor(kp22o,lp22o,w2/f);shading flat; hold on;
       [cc,hh]=contour(kp22o,lp22o,w2/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       title(['r=' num2str(r_2)],'Interpreter','latex','Fontsize',20);caxis(min_max2);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'xticklabel','','ytick',[0:.2:1]);
       set(gca,'yticklabel','');
   a6=axes('Position',[.55 .56 .4 .19]);
       pcolor(kp22o,lp22o,w2A/f);shading flat; hold on; caxis(min_max2);
       [cc,hh]=contour(kp22o,lp22o,w2A/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'xticklabel','','ytick',[0:.2:1]);
       set(gca,'yticklabel','');
   a7=axes('Position',[.55 .34 .4 .19]);
       pcolor(kp22o,lp22o,w2B/f); shading flat; hold on;caxis(min_max2);
       [cc,hh]=contour(kp22o,lp22o,w2B/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'ytick',[0:.2:1],'yticklabel','');
       set(gca,'xticklabel','');
   a8=axes('Position',[.55 .12 .4 .19]);
       pcolor(kp22o,lp22o,w2C/f);shading flat;hold on; caxis(min_max2);
       [cc,hh]=contour(kp22o,lp22o,w2C/f,'w','linewidth',1); clabel(cc,hh,'Fontsize',14);
       set(gca,'Fontsize',16,'xtick',[0:.2:1],'ytick',[0:.2:1],'yticklabel','');
       xlabel('k','Interpreter','latex');
       colorbar(a8,'south','Position',[.55 .03 .4 .02]);
       colormap(myc);

    print -dpng inertia_gravity_2D.png;close
