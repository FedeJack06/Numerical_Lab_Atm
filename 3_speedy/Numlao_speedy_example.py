#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:41:30 2022

@author: Paolo
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt


import cartopy.crs as ccrs
import cartopy

proj=ccrs.PlateCarree(central_longitude=180)
proj0=ccrs.PlateCarree(0)

seas='DJF'
DIR='/Users/Paolo/Downloads/SPEEDY_ENSO/' #path to directory with NetCDF files 
FILEclim=DIR+seas+'mean_clim.nc' #Full path with file name 

ncin=Dataset(FILEclim,'r')
print(ncin)
print(ncin.variables.keys())
print(ncin.variables['chi'])
print(ncin.variables['tp'])
chi=ncin.variables['chi'][:]
psi=ncin.variables['psi'][:]
u=ncin.variables['u'][:]
tp=ncin.variables['tp'][:]
temp0=ncin.variables['temp0'][:]
omega=ncin.variables['omega'][:]
lon=ncin.variables['lon'][:]
lat=ncin.variables['lat'][:]
lev=ncin.variables['lev'][:]
ncin.close()


#South America
las,lan=-25,-5
low,loe=290,320
#Europe
low,loe=0,40
las,lan=55,75
#Central America
low,loe=240,270
las,lan=20,40
fplot1=np.mean(temp0,axis=0)[:,:]
fplot2=np.mean(tp,axis=0)[:,:]
fig,ax=plt.subplots(subplot_kw=dict(projection=proj))
ax.contour(lon,lat,fplot1,colors='k',transform=proj0,levels=np.linspace(230,300,11),linewidths=1)
c=ax.contourf(lon,lat,fplot2,cmap='GnBu',transform=proj0,levels=np.linspace(0,10,11),extend='both')
#ax.contour(lon,lat,fplot2,colors='k',transform=proj0,levels=np.linspace(-150,150,15))
ax.coastlines(color='grey')
plt.colorbar(c,ax=ax,orientation='horizontal')
x=[low,loe,loe,low,low]
y=[las,las,lan,lan,las]
ax.plot(x,y,transform=proj0,color='red')
ax.set_xlim()
ax.set_title('T2m (temp0, K, contours) and TP (mm/day) DJF climatology ')
fig.savefig(seas+'_surface_ut.png',dpi=300)
plt.show()





