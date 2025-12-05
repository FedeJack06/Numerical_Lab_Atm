import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Variabile da analizzare
var_name = 'temp0'

# ============================================
# FILE 1: DJFmean_nino.nc
# ============================================
print("=== Analisi DJFmean_nino.nc ===")
ds_djf = xr.open_dataset('DJFmean_nino.nc')

pattern_correlation = []
for i in range(10):
    # Primo timestep
    first_djf = ds_djf[var_name].isel(time=i)

    # Media su tutti i 200 timesteps
    mean_djf = ds_djf[var_name].mean(dim='time')

    # Anomalia: primo timestep - media
    anomaly_djf = first_djf - mean_djf

    print(f"Timesteps totali: {len(ds_djf.time)}")
    print(f"Anomalia media globale: {float(anomaly_djf.mean()):.3f} K")
    print(f"Anomalia max: {float(anomaly_djf.max()):.3f} K")
    print(f"Anomalia min: {float(anomaly_djf.min()):.3f} K")

    # ============================================
    # FILE 2: JJAmean_nino.nc
    # ============================================
    print("\n=== Analisi JJAmean_nino.nc ===")
    ds_jja = xr.open_dataset('JJAmean_nino.nc')

    # Primo timestep
    first_jja = ds_jja[var_name].isel(time=0)

    # Media su tutti i 200 timesteps
    mean_jja = ds_jja[var_name].mean(dim='time')

    # Anomalia: primo timestep - media
    anomaly_jja = first_jja - mean_jja

    print(f"Timesteps totali: {len(ds_jja.time)}")
    print(f"Anomalia media globale: {float(anomaly_jja.mean()):.3f} K")
    print(f"Anomalia max: {float(anomaly_jja.max()):.3f} K")
    print(f"Anomalia min: {float(anomaly_jja.min()):.3f} K")

    pattern_correlation.append(anomaly_djf * anomaly_jja)
    
    #print(pattern_correlation)

    # ============================================
    # VISUALIZZAZIONE
    # ============================================

    
    # Crea figura con due subplot
    fig = plt.figure(figsize=(16, 14))


    # 3. CORR
    vmax = np.abs(pattern_correlation).max()
    vmin = -vmax
    ax3 = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())

    im3 = ax3.pcolormesh(ds_jja.lon, ds_jja.lat, pattern_correlation[i],
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', shading='auto',
                        vmin=vmin, vmax=vmax)

    ax3.coastlines()
    ax3.gridlines(alpha=0.3)
    ax3.set_global()

    plt.colorbar(im3, ax=ax3, orientation='horizontal', 
                pad=0.05, shrink=0.8, label=f'Anomalia {var_name} [K]')
    ax3.set_title(f'CORR anomalie')

    #plt.tight_layout()
    #plt.savefig('corr_anomalie_'+str(i)+'.png', dpi=150, bbox_inches='tight')
    #plt.show()

# 3. CORR
total_corr = pattern_correlation[0]
for i in range(8):
    total_corr = total_corr * pattern_correlation[i+1]
print(total_corr)
vmax = np.abs(total_corr).max()
vmin = -vmax
ax3 = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson())

im3 = ax3.pcolormesh(ds_jja.lon, ds_jja.lat, total_corr,
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r', shading='auto',
                    vmin=vmin, vmax=vmax)

ax3.coastlines()
ax3.gridlines(alpha=0.3)
ax3.set_global()

plt.colorbar(im3, ax=ax3, orientation='horizontal', 
            pad=0.05, shrink=0.8, label=f'Anomalia {var_name} [K]')
ax3.set_title(f'CORR TOT')

plt.tight_layout()
plt.savefig('corr_anomalie_.png', dpi=150, bbox_inches='tight')

# ============================================
# MAPPE SEPARATE (opzionale)
# ============================================

# Mappa solo DJF
'''fig, ax = plt.subplots(figsize=(14, 7),
                       subplot_kw={'projection': ccrs.Robinson()})

im = ax.pcolormesh(ds_djf.lon, ds_djf.lat, anomaly_djf,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto',
                   vmin=-np.abs(anomaly_djf).max(),
                   vmax=np.abs(anomaly_djf).max())

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'Anomalia {var_name} [K]')
ax.set_title(f'DJF: Anomalia primo timestep rispetto alla media climatologica')

plt.tight_layout()
plt.savefig('anomalia_djf.png', dpi=150, bbox_inches='tight')
plt.show()

# Mappa solo JJA
fig, ax = plt.subplots(figsize=(14, 7),
                       subplot_kw={'projection': ccrs.Robinson()})

im = ax.pcolormesh(ds_jja.lon, ds_jja.lat, anomaly_jja,
                   transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', shading='auto',
                   vmin=-np.abs(anomaly_jja).max(),
                   vmax=np.abs(anomaly_jja).max())

ax.coastlines()
ax.gridlines(alpha=0.3)
ax.set_global()

plt.colorbar(im, ax=ax, orientation='horizontal', 
             pad=0.05, shrink=0.8, label=f'Anomalia {var_name} [K]')
ax.set_title(f'JJA: Anomalia primo timestep rispetto alla media climatologica')

plt.tight_layout()
plt.savefig('anomalia_jja.png', dpi=150, bbox_inches='tight')
plt.show()

# Chiudi i dataset
ds_djf.close()
ds_jja.close()

print("\n=== File salvati ===")
print("- anomalie_primo_timestep.png (entrambe le mappe)")
print("- anomalia_djf.png (solo DJF)")
print("- anomalia_jja.png (solo JJA)")'''