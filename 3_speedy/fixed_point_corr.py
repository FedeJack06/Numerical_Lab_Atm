import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Configurazione per schermo 4K
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12

# Variabile da analizzare
var_name = 'temp0'
file = 'DJFmean_clim.nc'
ds_djf = xr.open_dataset(file)
da = ds_djf[var_name]
outDir = "out_img/fix_point_corr/"

# media temporale su tutti i 200 anni
mean_djf = da.mean(dim='time')

# ============================================
# 2. CALCOLO ANOMALIE
# ============================================
# time_i - MEDIA
anom_full = da - mean_djf

# ============================================
# 3. SELEZIONE DEL PUNTO DI RIFERIMENTO
# ============================================
# punto correlazione (Pacifico tropicale)
lat_target = 0.0
lon_target = -120.0

print(f"Punto scelto: Lat {lat_target}, Lon {lon_target}")

# serie di 200 anni del punto di rif
# method='nearest' trova il punto di griglia più vicino alle coordinate
ref_series = anom_full.sel(lat=lat_target, lon=lon_target, method='nearest')

# ============================================
# 4. CALCOLO DELLA CORRELAZIONE (One-Point Correlation)
# ============================================

# xr.corr calcola la correlazione di Pearson lungo la dimensione 'time'
# Confronta la serie del punto (ref_series) con OGNI altro punto della mappa (anom_full)
corr_map = xr.corr(anom_full, ref_series, dim='time')

# ============================================
# 5. PLOTTING
# ============================================
fig = plt.figure(figsize=(12, 7), constrained_layout=True)

ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax2.set_title(f"Mappa di Correlazione (Rif: {lat_target}N, {lon_target}E)", fontsize=14, weight='bold')
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

plt.savefig(outDir+'corr_'+file[0:3]+'-'+file[8:12]+'_'+f"{lat_target}N_{lon_target}E.png", dpi=150, bbox_inches='tight')
plt.show()