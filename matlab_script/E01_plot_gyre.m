% Simple script to vosualize GYRE exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/GYRE';
 model_re_t='GYRE_5d_00010101_00011230_grid_T.nc';
 model_re_u='GYRE_5d_00010101_00011230_grid_U.nc';
 model_re_v='GYRE_5d_00010101_00011230_grid_V.nc';
 model_re_w='GYRE_5d_00010101_00011230_grid_W.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'votemper');
 sal = ncread([model_path '/' model_re_t],'vosaline');
 ssh = ncread([model_path '/' model_re_t],'sossheig');
 uwf = ncread([model_path '/' model_re_t],'sowaflup');
 dhf = ncread([model_path '/' model_re_t],'sohefldo');
 zon = ncread([model_path '/' model_re_u],'vozocrtx');
 tax = ncread([model_path '/' model_re_u],'sozotaux');
 mer = ncread([model_path '/' model_re_v],'vomecrty');
 tay = ncread([model_path '/' model_re_v],'sometauy');

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 1000]);
   % Temp + current
   % --------------
   a=axes('Position',[0.1 0.55 .8 .4]);
   m_proj('albers equal-area','lat',[14.5 50],'long',[-78 -45],'rect','on');
   m_pcolor(lon,lat,tem(:,:,1,72));
   colormap(a,[m_colmap('jet',256)]);
   colorbar('Position',[.85 .55 .03 .38]);
   m_grid; hold on
   m_quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            zon(2:end-1,2:end-1,1,72),mer(2:end-1,2:end-1,1,72),5,'k');
   title('Surface Temp and current','interpreter','latex');
   set(gca,'Fontsize',18);
   % heat + stress
   % --------------
   b=axes('Position',[0.1 0.05 .8 .4]);
   m_proj('albers equal-area','lat',[14.5 50],'long',[-78 -45],'rect','on');
   m_pcolor(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1),dhf(2:end-1,2:end-1,72));
   colormap(b,[m_colmap('diverging',256)]);
   colorbar('Position',[.85 .06 .03 .38]);
   m_grid; hold on
   m_quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            tax(2:end-1,2:end-1,72),tay(2:end-1,2:end-1,72),2,'k');
   title('Heat and momentum fluxes','interpreter','latex');
   set(gca,'Fontsize',18);

print -dpng Ex1_figure.png;close;
