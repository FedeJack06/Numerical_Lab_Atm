import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Configurazione per schermo 4K
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12

# Variabile da analizzare e Nomi File
var_name = 'gh'
level = 1
season = 'DJF'
file_nino = season+'mean_nino.nc'
file_clim = season+'mean_clim.nc'
outDir = "out_img/anomaly/"

ds_nino = xr.open_dataset(file_nino)
ds_clim = xr.open_dataset(file_clim)

nino_var = ds_nino[var_name].isel(lev=level)
clim_var = ds_clim[var_name].isel(lev=level)

# media nino e clima della var
mean_var_clim = clim_var.mean(dim='time')
mean_var_nino = nino_var.mean(dim='time')

# 200 anomalie NINO
anom_var_nino = nino_var - mean_var_nino
# 200 anomalie CLIMA
anom_var_clim = clim_var - mean_var_clim

###########################################
# PLOTTING
time_step = 0
fig = plt.figure(figsize=(12, 14), constrained_layout=True)

ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson(central_longitude=0))
ax2.set_title(f"{season} - NINO", 
              fontsize=14, weight='bold')
ax2.coastlines()

# Plot corr
p2 = ax2.pcolormesh(anom_var_nino[time_step].lon, anom_var_nino[time_step].lat, anom_var_nino[time_step],
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r',
                    vmin=-np.abs(anom_var_nino[time_step]).max(), vmax=np.abs(anom_var_nino[time_step]).max())

ax2.legend(loc='lower left')

plt.colorbar(p2, ax=ax2, label='Anom gpm at 500 hPa', shrink=0.8)

############################
ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=0))
ax2.set_title(f"{season} - Clima", 
              fontsize=14, weight='bold')
ax2.coastlines()

# Plot corr
p2 = ax2.pcolormesh(anom_var_clim[time_step].lon, anom_var_clim[time_step].lat, anom_var_clim[time_step],
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r',
                    vmin=-np.abs(anom_var_clim[time_step]).max(), vmax=np.abs(anom_var_clim[time_step]).max())

ax2.legend(loc='lower left')

plt.colorbar(p2, ax=ax2, label='Anom gpm at 500 hPa', shrink=0.8)

# Nome file dinamico
nome_file_out = f"{outDir}{season}_500hPa_anom_t{time_step}.png"
plt.savefig(nome_file_out, dpi=200, bbox_inches='tight')

plt.show()