import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

#dir out
outDir = 'out_img/anomaly/'
# Carica i due file NetCDF
ds1 = xr.open_dataset('DJFmean_clim.nc')
ds2 = xr.open_dataset('DJFmean_nino.nc')

print(f"File 1 - timesteps: {len(ds1.time)}")
print(f"File 2 - timesteps: {len(ds2.time)}")

# Seleziona la variabile
var_name = 'temp0'

# Estrai i dati di temperatura dai due file
data1 = ds1[var_name]
data2 = ds2[var_name]

# Calcola la media temporale per ogni file
temp_mean1 = data1.mean(dim='time')
temp_mean2 = data2.mean(dim='time')

# Combina i due dataset lungo la dimensione temporale
data_combined = xr.concat([data1, data2], dim='time')
print(f"Timesteps totali combinati: {len(data_combined.time)}")

# Calcola la media temporale su tutti i 400 timesteps
temp_mean_total = data_combined.mean(dim='time')

# ============================================
# VISUALIZZAZIONE
# ============================================

# 1. Mappa della media totale (su tutti i 400 timesteps)
fig, ax = plt.subplots(figsize=(14, 7), dpi=200,
                       subplot_kw={'projection': ccrs.Robinson()})

im = ax.pcolormesh(ds1.lon, ds1.lat, temp_mean_total,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto')

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'{var_name} [K]')
ax.set_title(f'{var_name} - Media su 400 timesteps (clim + nino)')

plt.tight_layout()
plt.savefig(outDir+'media_totale_400timesteps.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Confronto tra le tre medie
'''fig = plt.figure(figsize=(18, 14), dpi=200)

# Media file clim
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.Robinson())
im1 = ax1.pcolormesh(ds1.lon, ds1.lat, temp_mean1,
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', shading='auto',
                     vmin=temp_mean_total.min(), 
                     vmax=temp_mean_total.max())
ax1.coastlines()
ax1.gridlines(alpha=0.3)
ax1.set_global()
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
ax1.set_title(f'{var_name} - Media CLIM (200 timesteps)')

# Media file nino
ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.Robinson())
im2 = ax2.pcolormesh(ds2.lon, ds2.lat, temp_mean2,
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', shading='auto',
                     vmin=temp_mean_total.min(), 
                     vmax=temp_mean_total.max())
ax2.coastlines()
ax2.gridlines(alpha=0.3)
ax2.set_global()
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
ax2.set_title(f'{var_name} - Media NINO (200 timesteps)')

# Media totale
ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson())
im3 = ax3.pcolormesh(ds1.lon, ds1.lat, temp_mean_total,
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', shading='auto',
                     vmin=temp_mean_total.min(), 
                     vmax=temp_mean_total.max())
ax3.coastlines()
ax3.gridlines(alpha=0.3)
ax3.set_global()
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
ax3.set_title(f'{var_name} - Media TOTALE (400 timesteps)')

plt.tight_layout()
plt.savefig(outDir+'confronto_medie.png', dpi=150, bbox_inches='tight')
plt.show()'''

# 3. Differenza tra NINO e CLIM
fig, ax = plt.subplots(figsize=(14, 7), dpi=200,
                       subplot_kw={'projection': ccrs.Robinson()})

diff = temp_mean2 - temp_mean1

im = ax.pcolormesh(ds1.lon, ds1.lat, diff,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto',
                   vmin=-np.abs(diff).max(),
                   vmax=np.abs(diff).max())

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'Δ{var_name} [K]')
ax.set_title(f'Differenza: NINO - CLIM')

plt.tight_layout()
plt.savefig(outDir+'differenza_nino_clim.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. NINO - MEDIA 400
fig, ax = plt.subplots(figsize=(14, 7), dpi=200,
                       subplot_kw={'projection': ccrs.Robinson()})

diff_nino_total = temp_mean2 - temp_mean_total

im = ax.pcolormesh(ds1.lon, ds1.lat, diff_nino_total,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto',
                   vmin=-np.abs(diff_nino_total).max(),
                   vmax=np.abs(diff_nino_total).max())

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'Δ{var_name} [K]')
ax.set_title(f'Differenza: NINO - MEDIA 400 timesteps')

plt.tight_layout()
plt.savefig(outDir+'differenza_nino_media400.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. CLIM - MEDIA 400
fig, ax = plt.subplots(figsize=(14, 7), dpi=200,
                       subplot_kw={'projection': ccrs.Robinson()})

diff_clim_total = temp_mean1 - temp_mean_total

im = ax.pcolormesh(ds1.lon, ds1.lat, diff_clim_total,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto',
                   vmin=-np.abs(diff_clim_total).max(),
                   vmax=np.abs(diff_clim_total).max())

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'Δ{var_name} [K]')
ax.set_title(f'Differenza: CLIM - MEDIA 400 timesteps')

plt.tight_layout()
plt.savefig(outDir+'differenza_clim_media400.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. CLIM con contour della differenza NINO-CLIM sovrapposti
fig, ax = plt.subplots(figsize=(14, 7), dpi=200,
                       subplot_kw={'projection': ccrs.Robinson()})

# Mappa a colori: media CLIM
im = ax.pcolormesh(ds1.lon, ds1.lat, temp_mean1,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto')

# Contour neri: differenza NINO - CLIM
diff_nino_clim = temp_mean2 - temp_mean1
contours = ax.contour(ds1.lon, ds1.lat, diff_nino_clim,
                      levels=10,
                      colors='black',
                      linewidths=1.5,
                      transform=ccrs.PlateCarree())

# Etichette sui contour
ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'{var_name} CLIM [K]')
ax.set_title(f'{var_name} CLIM (colori) + Differenza NINO-CLIM (contour neri)')

plt.tight_layout()
plt.savefig(outDir+'clim_con_contour_diff.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistiche
print("\n=== STATISTICHE ===")
print(f"Media CLIM: {float(temp_mean1.mean()):.2f} K")
print(f"Media NINO: {float(temp_mean2.mean()):.2f} K")
print(f"Media TOTALE: {float(temp_mean_total.mean()):.2f} K")
print(f"\nDifferenza media (NINO-CLIM): {float((temp_mean2-temp_mean1).mean()):.2f} K")
print(f"Differenza media (NINO-MEDIA400): {float((temp_mean2-temp_mean_total).mean()):.2f} K")
print(f"Differenza media (CLIM-MEDIA400): {float((temp_mean1-temp_mean_total).mean()):.2f} K")

# Chiudi i dataset
ds1.close()
ds2.close()