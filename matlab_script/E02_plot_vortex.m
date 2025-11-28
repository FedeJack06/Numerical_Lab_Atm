% Simple script to visualize VORTEX exercise results
% assumes you instaled m_map
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/VORTEX';
 model_re_t='VORTEX_grid_T.nc';
 model_re_u='VORTEX_grid_U.nc';
 model_re_v='VORTEX_grid_V.nc';
 model_re_1='1_VORTEX_grid_W.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');
 lo1 = ncread([model_path '/' model_re_1],'nav_lon');
 la1 = ncread([model_path '/' model_re_1],'nav_lat');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'thetao_inst');
 zon = ncread([model_path '/' model_re_u],'uo_inst');
 mer = ncread([model_path '/' model_re_v],'vo_inst');

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 600]);
   % Temp + current (time 1)
   % --------------
   a=axes('Position',[0.1 0.55 .4 .4]);
   pcolor(lon(4:end-4,4:end-4),lat(4:end-4,4:end-4), ...
          tem(4:end-4,4:end-4,1,1));shading flat; hold on;
   line(double(lo1(1,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(end,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(:,1)),double(la1(:,1)),'linewidth',2,'color','k');
   line(double(lo1(:,end)),double(la1(:,end)),'linewidth',2,'color','k');
   quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            zon(2:end-1,2:end-1,1,1),mer(2:end-1,2:end-1,1,1),5,'k');
   title('Surface T and Vel (Init)','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');
caxis([19.1 20.5]);
%
   a=axes('Position',[0.55 0.55 .4 .4]);
   pcolor(lon(4:end-4,4:end-4),lat(4:end-4,4:end-4), ...
          tem(4:end-4,4:end-4,1,25));shading flat; hold on;
   colormap(a,[m_colmap('jet',256)]);
   line(double(lo1(1,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(end,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(:,1)),double(la1(:,1)),'linewidth',2,'color','k');
   line(double(lo1(:,end)),double(la1(:,end)),'linewidth',2,'color','k');
   quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            zon(2:end-1,2:end-1,1,25),mer(2:end-1,2:end-1,1,25),5,'k');
   title('1/4','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','','Xticklabel','');caxis([19.1 20.5]);
%
   a=axes('Position',[0.1 0.1 .4 .4]);
   pcolor(lon(4:end-4,4:end-4),lat(4:end-4,4:end-4), ...
          tem(4:end-4,4:end-4,1,50));shading flat; hold on;
   colormap(a,[m_colmap('jet',256)]);
   line(double(lo1(1,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(end,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(:,1)),double(la1(:,1)),'linewidth',2,'color','k');
   line(double(lo1(:,end)),double(la1(:,end)),'linewidth',2,'color','k');
   quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            zon(2:end-1,2:end-1,1,50),mer(2:end-1,2:end-1,1,50),5,'k');
   title('1/2','interpreter','latex');
   set(gca,'Fontsize',18);caxis([19.1 20.5]);
%
   a=axes('Position',[0.55 0.1 .4 .4]);
   pcolor(lon(4:end-4,4:end-4),lat(4:end-4,4:end-4), ...
          tem(4:end-4,4:end-4,1,end));shading flat; hold on;
   colormap(a,[m_colmap('jet',256)]);
   line(double(lo1(1,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(end,:)),double(la1(1,:)),'linewidth',2,'color','k');
   line(double(lo1(:,1)),double(la1(:,1)),'linewidth',2,'color','k');
   line(double(lo1(:,end)),double(la1(:,end)),'linewidth',2,'color','k');
   quiver(lon(2:end-1,2:end-1),lat(2:end-1,2:end-1), ...
            zon(2:end-1,2:end-1,1,end),mer(2:end-1,2:end-1,1,end),5,'k');
   title('End','interpreter','latex');
   set(gca,'Fontsize',18,'Yticklabel','');caxis([19.1 20.5]);

   colormap([m_colmap('jet',256)]);
   colorbar('horizontal','Position',[.15 .035 .75 .02]);
%
print -dpng Ex2_figure.png;close;
