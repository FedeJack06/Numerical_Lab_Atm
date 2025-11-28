% Simple script to visualize LOCK OVERFLOW exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/EQ-WAVES';
 model_re_t='EQ-WAVE_7d_00010101_00010720_grid_T.nc';
 model_re_u='EQ-WAVE_7d_00010101_00010720_grid_U.nc';
 model_re_v='EQ-WAVE_7d_00010101_00010720_grid_V.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');
 dpt = ncread([model_path '/' model_re_t],'deptht');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'votemper');
 tax = ncread([model_path '/' model_re_u],'sozotaux');
 tay = ncread([model_path '/' model_re_v],'sometauy');

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 1000]);
   % --------------
   a=axes('Position',[0.1 0.58 .8 .5]);
   m_proj('miller','lon',[147 262],'lat',[-22 22]);
   m_pcolor(lon,lat,squeeze(tem(:,:,1,1)));shading flat; hold on;
   title('T Wind (Init)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);
   m_quiver(lon(2:30:end-1,2:10:end-1)  ,lat(2:30:end-1,2:10:end-1), ...
            tax(2:30:end-1,2:10:end-1,1),tay(2:30:end-1,2:10:end-1,1), ...
            'color','w','linewidth',2);
   m_grid
   caxis([5 21]);
%
   b=axes('Position',[0.1 0.38 .8 .25]);
   pcolor(lon(:,1),-dpt,squeeze(tem(:,41,:,1))');shading flat; hold on;
   contour(lon(:,1),-dpt,squeeze(tem(:,41,:,1))','w');shading flat; hold on;
   title('Equatorial Section (Init)','interpreter','latex');
   set(gca,'Fontsize',18); %,'Xticklabel','','Yticklabel','');
   colormap(b,[m_colmap('jet',256)]); %caxis([-5*10^-5 5*10^-5]);
   caxis([5 21]);
%
   tt=8;
   a=axes('Position',[0.1 -.07 .8 .5]);
   m_proj('miller','lon',[147 262],'lat',[-22 22]);
   m_pcolor(lon,lat,squeeze(tem(:,:,1,tt)));shading flat; hold on;
   title('T Wind (Perturb)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
   colormap(a,[m_colmap('jet',256)]);
   m_quiver(lon(2:30:end-1,2:10:end-1)  ,lat(2:30:end-1,2:10:end-1), ...
            tax(2:30:end-1,2:10:end-1,tt),tay(2:30:end-1,2:10:end-1,tt), ...
            'color','w','linewidth',2);
   m_grid
   caxis([5 21]);
   colorbar(a,'Position',[.92 .15 .02 .7]);
%%%
print -dpng Ex7_figure_01.png;close;
