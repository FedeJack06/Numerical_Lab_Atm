import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Configurazione per schermo 4K
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12

# Variabile da analizzare e Nomi File
var_name = 'temp0'
season = 'DJF'
file_nino = season+'mean_nino.nc'
file_clim = season+'mean_clim.nc'
outDir = "out_img/anomaly/"

ds_nino = xr.open_dataset(file_nino)
ds_clim = xr.open_dataset(file_clim)

nino_temp = ds_nino['temp0']
clim_temp = ds_clim['temp0']

nino_ah = ds_nino['q'].isel(lev=0)
clim_ah = ds_clim['q'].isel(lev=0)

# media CLIMA
mean_temp_clima = clim_temp.mean(dim='time')
mean_ah_clima = clim_ah.mean(dim='time')

# anomalie NINO - media CLIMA
anom_temp = nino_temp - mean_temp_clima
anom_ah = nino_ah - mean_ah_clima

# remove noise
signal_temp = anom_temp.mean(dim='time')
signal_ah = anom_ah.mean(dim='time')

###########################################
# PLOTTING
fig = plt.figure(figsize=(12, 14), constrained_layout=True)

ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson(central_longitude=150))
ax2.set_title(f"{season} temp - mean anomaly - media clima", 
              fontsize=14, weight='bold')
ax2.coastlines()

# Plot corr
p2 = ax2.pcolormesh(signal_temp.lon, signal_temp.lat, signal_temp,
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r',
                    vmin=-1, vmax=1)

ax2.legend(loc='lower left')

plt.colorbar(p2, ax=ax2, label='Coefficiente di Correlazione', shrink=0.8)

############################
ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=150))
ax2.set_title(f"{season} AH - mean anomaly - media clima", 
              fontsize=14, weight='bold')
ax2.coastlines()

# Plot corr
p2 = ax2.pcolormesh(signal_ah.lon, signal_ah.lat, signal_ah,
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r',
                    vmin=-1, vmax=1)

ax2.legend(loc='lower left')

plt.colorbar(p2, ax=ax2, label='Coefficiente di Correlazione', shrink=0.8)

# Nome file dinamico
nome_file_out = f"{outDir}signal_{season}.png"
plt.savefig(nome_file_out, dpi=200, bbox_inches='tight')
print(f"Salvato: {nome_file_out}")

plt.show()

# CONTROLLO VARIANZA
'''std_dev = anom_full.std(dim='time')

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
std_dev.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Std Dev (K)'})
ax.coastlines()
ax.plot(lon_target, lat_target, 'g*', markersize=10)
plt.title("Deviazione Standard (Combined Clim + Nino)")
plt.show()

# CONTROLLO NUMERICO
std_point = std_dev.sel(lat=lat_target, lon=lon_target, method='nearest').item()
print(f"Deviazione standard nel punto scelto (deve essere > 0): {std_point:.4f}")'''