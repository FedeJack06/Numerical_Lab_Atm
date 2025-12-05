import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Configurazione per schermo 4K
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12

# Variabile da analizzare e Nomi File
var_name = 'temp0'
file_nino = 'DJFmean_nino.nc'
file_clim = 'DJFmean_clim.nc'
outDir = "out_img/fix_point_corr/"

ds_nino = xr.open_dataset(file_nino)
ds_clim = xr.open_dataset(file_clim)

# concatenazione Climatologia + Nino
da = xr.concat([ds_clim[var_name], ds_nino[var_name]], dim='time')

# media su tutto
mean_total = da.mean(dim='time')

# Anomalia rispetto alla media totale
anom_full = da - mean_total

################################################
# PUNTO DI RIFERIMENTO
lat_target = 0.0
lon_target = -120.0

print(f"Punto scelto: Lat {lat_target}, Lon {lon_target}")

# Estrazione serie temporale del punto (che ora contiene il salto Clim -> Nino)
ref_series = anom_full.sel(lat=lat_target, lon=lon_target, method='nearest')

#############################################
# One-Point Correlation

print("Calcolo correlazione...")
# Confronta il "salto" nel punto rif con il "salto" in ogni altro punto
corr_map = xr.corr(anom_full, ref_series, dim='time')

###########################################
# PLOTTING
fig = plt.figure(figsize=(12, 7), constrained_layout=True)

ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax2.set_title(f"Teleconnessioni (Correlazione Clim+Nino) - Rif: {lat_target}N, {lon_target}E", 
              fontsize=14, weight='bold')
ax2.coastlines()

# Plot corr
p2 = ax2.pcolormesh(corr_map.lon, corr_map.lat, corr_map,
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r',
                    vmin=-1, vmax=1)

ax2.plot(lon_target, lat_target, marker='*', color='green', markersize=15, 
         transform=ccrs.PlateCarree(), markeredgecolor='black', label='Punto Riferimento')
ax2.legend(loc='lower left')

plt.colorbar(p2, ax=ax2, label='Coefficiente di Correlazione', shrink=0.8)

# Nome file dinamico
nome_file_out = f"{outDir}corr_combined_{lat_target}N_{lon_target}E.png"
plt.savefig(nome_file_out, dpi=150, bbox_inches='tight')
print(f"Salvato: {nome_file_out}")

# CONTROLLO VARIANZA
std_dev = anom_full.std(dim='time')

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
std_dev.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Std Dev (K)'})
ax.coastlines()
ax.plot(lon_target, lat_target, 'g*', markersize=10)
plt.title("Deviazione Standard (Combined Clim + Nino)")
plt.show()

# CONTROLLO NUMERICO
std_point = std_dev.sel(lat=lat_target, lon=lon_target, method='nearest').item()
print(f"Deviazione standard nel punto scelto (deve essere > 0): {std_point:.4f}")