% Simple script to visualize LOCK OVERFLOW exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/EQ-WAVES';
 model_re_t='EQ-WAVE_7d_00010101_00010720_grid_T.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');
 dpt = ncread([model_path '/' model_re_t],'deptht');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'votemper');

% remove zeros (?)
  tem(tem==0.0)=nan;

  tr  = 8; % ( pertubation reference)
  jpk = 12;% 37.50 m
  mi=-5;
  ma=5;
% ----------------
 figure('Position',[10 10 600 1000]);
   % --------------
   tt=8;
   a=axes('Position',[0.1 0.75 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)));shading flat; hold on;
   title('T [287m]','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);
   grid on;caxis([5 20]);
%
   b=axes('Position',[0.55 0.75 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)-tem(:,:,jpk,tr)));shading flat; hold on;
   title('T anomaly (after pert)','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','','Xticklabel','');
   colormap(b,[m_colmap('diverging',256)]);
   caxis([mi ma]);grid on

   tt=9;
   a=axes('Position',[0.1 0.53 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)));shading flat; hold on;
   title('T day=10 aft Pert','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);
   grid on;caxis([5 20]);

   b=axes('Position',[0.55 0.53 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)-tem(:,:,jpk,tr)));shading flat; hold on;
   title('T anomaly ','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','','Xticklabel','');
   colormap(b,[m_colmap('diverging',256)]);
   caxis([mi ma]);grid on

   tt=11;
   a=axes('Position',[0.1 0.3 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)));shading flat; hold on;
   title('T day=30 aft Pert','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);
   grid on;caxis([5 20]);

   b=axes('Position',[0.55 0.3 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)-tem(:,:,jpk,tr)));shading flat; hold on;
   title('T anomaly (30 days)','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','','Xticklabel','');
   colormap(b,[m_colmap('diverging',256)]);
   caxis([mi ma]);grid on

   tt=14;
   a=axes('Position',[0.1 0.07 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)));shading flat; hold on;
   title('T day=60 after pert','interpreter','latex');
   set(gca,'Fontsize',18);
   colormap(a,[m_colmap('jet',256)]);
   grid on;caxis([5 20]);

   b=axes('Position',[0.55 0.07 .4 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,jpk,tt)-tem(:,:,jpk,tr)));shading flat; hold on;
   title('T anomaly (60 days)','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','');
   colormap(b,[m_colmap('diverging',256)]);
   caxis([mi ma]);grid on
%
print -dpng Ex7_figure_02.png;close;
