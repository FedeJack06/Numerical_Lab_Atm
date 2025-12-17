"""
Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector during winter time.

This example uses the metadata-retaining xarray interface.

Additional requirements for this example:

    * xarray (http://xarray.pydata.org)
    * matplotlib (http://matplotlib.org/)
    * cartopy (http://scitools.org.uk/cartopy/)

"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr

from eofs.xarray import Eof
from eofs.examples import example_data_path

# Read geopotential height data using the xarray module. The file contains
# December-February averages of geopotential height at 500 hPa for the
# European/Atlantic domain (80W-40E, 20-90N).
#filename = example_data_path('hgt_djf.nc')
filename = 'DJFmean_clim.nc'
title = filename[:3]+" - "+filename[8:12]
level = 2 # 500 hPa
modo = 0
timestep = 0
z_djf = xr.open_dataset(filename)['gh']

# Compute anomalies by removing the time-mean.
z_djf = z_djf - z_djf.mean(dim='time')

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_djf.coords['lat'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_djf, weights=wgts)

# Retrieve the leading EOF, expressed as the covariance between the leading PC
# time series and the input SLP anomalies at each grid point.
eof1 = solver.eofsAsCovariance(neofs=2)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
'''clevs = np.linspace(-75, 75, 11)
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_global()
#modo 0 e livello 2 500 hPa
eo = eof1.isel(mode=modo, lev=level).plot.contourf(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                         transform=ccrs.PlateCarree(), add_colorbar=False)
plt.colorbar(eo, ax=ax, label='', shrink=0.8)
ax.set_title('EOF1 expressed as covariance', fontsize=16)

###################################################### index
anomaly = z_djf.isel(time=timestep, lev=level)
indice = eof1.isel(mode=modo, lev=level) * anomaly

max_index = np.abs(indice).max().item()
print(indice.sizes)

#plot
fetta_zero = indice.isel(lon=0)
fetta_360 = fetta_zero.assign_coords(lon=360)
indice = xr.concat([indice, fetta_360], dim='lon')

plt.figure(2)
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax2 = plt.axes(projection=proj)
ax2.coastlines()
ax2.set_global()
ind = indice.plot.contourf(ax=ax2, levels=np.linspace(-200, 200, 11),
                         cmap=plt.cm.RdBu_r,
                         transform=ccrs.PlateCarree(), add_colorbar=False)
plt.colorbar(ind, ax=ax2, label='index', shrink=0.8)
ax2.set_title(f'Indice NAO DJF timestep {timestep}', fontsize=16)'''

####################################################
nao_i = []
for i in range( len(z_djf.coords['time'].values) ):
    anomaly = z_djf.isel(time=i, lev=level)
    index_spatial =  eof1.isel(mode=modo, lev=level) * anomaly
    nao_i.append(index_spatial.sum())

'''fig, ax = plt.subplots(3,1,figsize=(18,12))
ax[0].plot(x, nao_i, color='black', alpha=0)
ax[0].fill_between(x, nao_i, 0, 
                 where=(np.array(nao_i) >= 0),    # Condizione: y maggiore o uguale a 0
                 color='red', 
                 alpha=1,         # Trasparenza
                 interpolate=True)  # Fondamentale per incroci puliti

# 4. Riempimento BLU per i valori sotto lo zero
ax[0].fill_between(x, nao_i, 0, 
                 where=(np.array(nao_i) <= 0),    # Condizione: y minore o uguale a 0
                 color='blue', 
                 alpha=1, 
                 interpolate=True)
ax[0].axhline(0, color='black', linewidth=1)
ax[0].grid()
ax[0].set_title("NINO JJA")
ax[0].set_xlabel("Time step")
ax[0].set_ylabel("NAO index")'''

x = list(range(len(nao_i)))
num_plots = 4
chunk_size = 50
fig, axs = plt.subplots(num_plots, 1, figsize=(18, 12), sharey=True)

for i, ax in enumerate(axs):
    start = i * chunk_size
    end = (i+1) * chunk_size
    
    x_slice = x[start:end]
    y_slice = nao_i[start:end]

    ax.plot(x_slice, y_slice, color='black', alpha=0)
    ax.fill_between(x_slice, y_slice, 0, 
                 where=(np.array(y_slice) >= 0),
                 color='red', 
                 alpha=1,
                 interpolate=True)
    ax.fill_between(x_slice, y_slice, 0, 
                    where=(np.array(y_slice) <= 0),
                    color='blue', 
                    alpha=1, 
                    interpolate=True)
    ax.axhline(0, color='black', linewidth=1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.3)

    ax.set_xlabel("Time step")
    ax.set_ylabel("NAO index")
axs[0].set_title(title)
#plt.savefig("out_img/nao/")
plt.show()