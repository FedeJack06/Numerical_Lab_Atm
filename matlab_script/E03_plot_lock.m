% Simple script to visualize LOCK EXCHANGE exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/LOCK_EXCHANGE';
 model_re_t='LOCK_FCT2_flux_ubs_grid_T.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');
 dpt = ncread([model_path '/' model_re_t],'deptht');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'thetao_inst');

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 1000]);
   % Temp + current (time 1)
   % --------------
   a=axes('Position',[0.1 0.75 .8 .18]);
   pcolor(lon(:,1),-dpt,squeeze(tem(:,2,:,1))');shading flat; hold on;
   title('T cross section (Init)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);caxis([5 29]);
%
   a=axes('Position',[0.1 0.52 .8 .18]);
   pcolor(lon(:,1),-dpt,squeeze(tem(:,2,:,14))');shading flat; hold on;
   title('1/4','interpreter','latex');
   colormap(a,[m_colmap('jet',256)]);
   set(gca,'Fontsize',18,'Xticklabel','');caxis([5 29]);
%
   a=axes('Position',[0.1 0.29 .8 .18]);
   pcolor(lon(:,1),-dpt,squeeze(tem(:,2,:,24))');shading flat; hold on;
   title('1/2','interpreter','latex');
   colormap(a,[m_colmap('jet',256)]);
   set(gca,'Fontsize',18,'Xticklabel','');caxis([5 29]);
%
   a=axes('Position',[0.1 0.07 .8 .18]);
   pcolor(lon(:,1),-dpt,squeeze(tem(:,2,:,34))');shading flat; hold on;
   colormap(a,[m_colmap('jet',256)]);
   title('End','interpreter','latex');
   set(gca,'Fontsize',18);caxis([5 29]);

   colorbar('Position',[.92 .15 .02 .7]);
%
print -dpng Ex3_figure.png;close;
