% Simple script to visualize LOCK OVERFLOW exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/CANAL';
 model_re_t='CANAL_grid_T.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon_grid_T');
 lat = ncread([model_path '/' model_re_t],'nav_lat_grid_T');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'toce');
 rvo = ncread([model_path '/' model_re_t],'ssrelvor');

 [d1,d2,d3,d4]=size(tem)
 %
% Divide time dimension to plot 4 equally distand snapshots
  delta_4=floor(d4/3);
  plot_t=[1:delta_4:d4];

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 1000]);
   % --------------
   a=axes('Position',[0.1 0.75 .35 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,1,plot_t(1))));shading flat; hold on;
   contour(lon,lat,squeeze(tem(:,:,1,plot_t(1))),'w');
   title('T (Init)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);caxis([3 14]);

   b=axes('Position',[0.55 0.75 .35 .18]);
   pcolor(lon,lat,squeeze(rvo(:,:,plot_t(1))));shading flat; hold on;
   contour(lon,lat,squeeze(rvo(:,:,plot_t(1))),'w');
   title('Rel Vor (Init)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','','Yticklabel','');
   colormap(b,[m_colmap('jet',256)]);caxis([-5*10^-5 5*10^-5]);
%
   a=axes('Position',[0.1 0.52 .35 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,1,plot_t(2))));shading flat; hold on;
   contour(lon,lat,squeeze(tem(:,:,1,plot_t(2))),'w');
   title([' kt=' num2str(plot_t(2))],'interpreter','latex');
   colormap(a,[m_colmap('jet',256)]);caxis([3 14]);
   set(gca,'Fontsize',18,'Xticklabel','');

   b=axes('Position',[0.55 0.52 .35 .18]);
   pcolor(lon,lat,squeeze(rvo(:,:,plot_t(2))));shading flat; hold on;
   contour(lon,lat,squeeze(rvo(:,:,plot_t(2))),'w');
   title([' kt=' num2str(plot_t(2))],'interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','','Yticklabel','');
   colormap(b,[m_colmap('jet',256)]);caxis([-5*10^-5 5*10^-5]);
%
   a=axes('Position',[0.1 0.29 .35 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,1,plot_t(3))));shading flat; hold on;
   contour(lon,lat,squeeze(tem(:,:,1,plot_t(3))),'w');
   title([' kt=' num2str(plot_t(3))],'interpreter','latex');
   colormap(a,[m_colmap('jet',256)]);
   set(gca,'Fontsize',18,'Xticklabel','');caxis([3 14]);

   b=axes('Position',[0.55 0.29 .35 .18]);
   pcolor(lon,lat,squeeze(rvo(:,:,plot_t(3))));shading flat; hold on;
   contour(lon,lat,squeeze(rvo(:,:,plot_t(3))),'w');
   title([' kt=' num2str(plot_t(3))],'interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','','Yticklabel','');
   colormap(b,[m_colmap('jet',256)]);caxis([-5*10^-5 5*10^-5]);
%%
   a=axes('Position',[0.1 0.07 .35 .18]);
   pcolor(lon,lat,squeeze(tem(:,:,1,plot_t(4))));shading flat; hold on;
   contour(lon,lat,squeeze(tem(:,:,1,plot_t(4))),'w');
   colormap(a,[m_colmap('jet',256)]);
   title('End','interpreter','latex');
   set(gca,'Fontsize',18);caxis([3 14]);
%
   b=axes('Position',[0.55 0.07 .35 .18]);
   pcolor(lon,lat,squeeze(rvo(:,:,plot_t(4))));shading flat; hold on;
   contour(lon,lat,squeeze(rvo(:,:,plot_t(4))),'w');
   title('End','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','');
   colormap(b,[m_colmap('jet',256)]);caxis([-5*10^-5 5*10^-5]);

   colorbar(a,'Position',[.455 .15 .02 .7]);
   colorbar(b,'Position',[.905 .15 .02 .7]);
%%
print -dpng Ex6_figure.png;close;
