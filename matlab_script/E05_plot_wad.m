% Simple script to visualize WAD exercise results
% P. Oddo
% --------------------------------------------------
 model_path='/Users/poddo/Courses/Numerical_Ocean_lab/Exercises/model_results/WAD';
 model_re_t='WAD_1h_00010101_00010101_grid_T.nc';
 model_mesh='mesh_mask.nc';

% Load coordinates
% ----------------
 lon = ncread([model_path '/' model_re_t],'nav_lon');
 lat = ncread([model_path '/' model_re_t],'nav_lat');
 dpt = ncread([model_path '/' model_re_t],'deptht');
 bty = ncread([model_path '/' model_mesh],'ht_0');

% Load model results
% ----------------
 tem = ncread([model_path '/' model_re_t],'votemper');
 ssh = ncread([model_path '/' model_re_t],'sossheig');

% remove zeros (?)
  tem(tem==0.0)=nan;
% plot
% ----------------
 figure('Position',[10 10 600 1000]);
   % SSH
   % --------------
    a1=[lon(2:end-1,12)' flipud(lon(2:end-1,12))'];
    b1=[-bty(2:end-1,12)' ones(1,size(bty,1)-3)*(-bty(end-1,12)) -bty(2,12)];

   a=axes('Position',[0.1 0.75 .8 .18]);
    c1=[ssh(2:end-1,12,1)' ones(1,size(bty,1)-3)*(-bty(end-1,12)) ssh(2,12,1)];
    patch(a1,c1,[.0 .0 .8]);hold on
    patch(a1,b1,[.8 .8 .8])
    xlim([2.5 48]);ylim([-8 .5]);
    title('SSH cross section (Init)','interpreter','latex');
    set(gca,'Fontsize',18,'Xticklabel','');
%
   a=axes('Position',[0.1 0.52 .8 .18]);
    c1=[ssh(2:end-1,12,2)' ones(1,size(bty,1)-3)*(-bty(end-1,12)) ssh(2,12,2)];
    patch(a1,c1,[.0 .0 .8]);hold on
    patch(a1,b1,[.8 .8 .8])
    xlim([2.5 48]);ylim([-8 .5]);
    set(gca,'Fontsize',18,'Xticklabel','');
    title('1h','interpreter','latex');
%
   a=axes('Position',[0.1 0.29 .8 .18]);
    c1=[ssh(2:end-1,12,3)' ones(1,size(bty,1)-3)*(-bty(end-1,12)) ssh(2,12,3)];
    patch(a1,c1,[.0 .0 .8]);hold on
    patch(a1,b1,[.8 .8 .8])
    xlim([2.5 48]);ylim([-8 .5]);
    set(gca,'Fontsize',18,'Xticklabel','');
   title('2h','interpreter','latex');
   set(gca,'Fontsize',18,'Xticklabel','');caxis([10 22]);
%%
   a=axes('Position',[0.1 0.07 .8 .18]);
    c1=[ssh(2:end-1,12,4)' ones(1,size(bty,1)-3)*(-bty(end-1,12)) ssh(2,12,4)];
    patch(a1,c1,[.0 .0 .8]);hold on
    patch(a1,b1,[.8 .8 .8])
    xlim([2.5 48]);ylim([-8 .5]);
   title('3h','interpreter','latex');
   set(gca,'Fontsize',18);caxis([10 22]);
%
%
print -dpng Ex5_figure.png;close;
