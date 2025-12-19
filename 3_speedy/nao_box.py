import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr

from eofs.xarray import Eof
from eofs.examples import example_data_path

import os

# Ottieni il percorso assoluto della directory dello script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Dizionario dei file da processare
files = {
    'Clim': os.path.join(script_dir, 'DJFmean_clim.nc'),
    'Nino': os.path.join(script_dir, 'DJFmean_nino.nc')
}

# Dizionario per salvare gli indici calcolati
nao_indices = {}

level = 2 # 500 hPa
modo = 0

# --- Ciclo di elaborazione per entrambi i casi (Clim e Nino) ---
for label, filename in files.items():
    print(f"Processing: {label} ({filename})")
    
    # 1. Caricamento Dati
    z_djf = xr.open_dataset(filename)['gh']
    z_djf = z_djf.sel(lat=slice(20, 90))

    # 2. Calcolo Anomalie
    z_djf = z_djf - z_djf.mean(dim='time')

    # 3. Solver EOF
    coslat = np.cos(np.deg2rad(z_djf.coords['lat'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(z_djf, weights=wgts)

    # 4. Estrazione Leading EOF (Pattern Spaziale)
    eof1 = solver.eofsAsCovariance(neofs=2)
    
    # Pattern spaziale fisso per la proiezione
    spatial_pattern = eof1.isel(mode=modo, lev=level)

    # 5. Calcolo Indice NAO (Proiezione manuale come da tuo script)
    nao_i = []
    # Loop sui timestep
    for i in range(len(z_djf.coords['time'].values)):
        anomaly = z_djf.isel(time=i, lev=level)
        # Proiezione: pattern * anomalia
        index_spatial = spatial_pattern * anomaly
        # Somma spaziale per ottenere lo scalare (PC value)
        nao_i.append(index_spatial.sum())

    # 6. Standardizzazione dell'indice
    nao_i = np.array(nao_i)
    mean_nao = np.mean(nao_i)
    dev_nao = np.std(nao_i)
    nao_standardized = (nao_i - mean_nao) / dev_nao
    
    # Salvataggio nel dizionario
    nao_indices[label] = nao_standardized
    print(f" -> {label} processed. Mean: {mean_nao:.2f}, Std: {dev_nao:.2f}")


# --- Creazione del Boxplot (Whisker Plot) ---
print("\nGenerating Boxplot...")

# Riduco la larghezza della figura (figsize) per ridurre lo spazio bianco laterale
fig, ax = plt.subplots(figsize=(16, 9))

data_to_plot = [nao_indices['Clim'], nao_indices['Nino']]
labels = ['Clim', 'Niño']

# 1. Avvicinare i box: definiamo posizioni manuali vicine (es. 1.0 e 1.7)
# Default sarebbe [1, 2]. Width controlla la larghezza del box stesso.
pos = [1, 1.7]
width = 0.3 # Aumento leggermente la larghezza per proporzione

bplot = ax.boxplot(data_to_plot, 
                   positions=pos,
                   widths=width,
                   tick_labels=labels,
                   showmeans=True,
                   meanline=True,  # Mostra la media come una linea (barra) invece che un punto
                   patch_artist=True,
                   medianprops=dict(color="black", linewidth=1.5),
                   meanprops=dict(linestyle='--', linewidth=1.5, color='green'))

# Personalizzazione colori
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 2. Aggiungere i valori numerici vicino ai quartili
# Iteriamo sui dati originali per calcolare le statistiche esatte
for i, data in enumerate(data_to_plot):
    # Calcolo statistiche
    q1 = np.percentile(data, 25)
    med = np.median(data)
    q3 = np.percentile(data, 75)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    # Coordinata X per il testo: posizione del box + metà larghezza + piccolo margine
    x_text = pos[i] + (width / 2) + 0.05
    
    # Aggiunta testo sul grafico (va='center' allinea verticalmente al punto)
    ax.text(x_text, q1, f'Q1: {q1:.2f}', va='center', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(x_text, med + 0.05, f'Median: {med:.2f}', va='center', fontsize=12, fontweight='bold', color='black')
    ax.text(x_text, q3, f'Q3: {q3:.2f}', va='center', fontsize=12, fontweight='bold', color='darkred')
    ax.text(x_text, mean_val - 0.05, rf'Mean: {mean_val:.2f} $\pm$ {std_val:.2f}', va='center', fontsize=12, fontweight='bold', color='green')

# Titoli e griglia
#ax.set_title('NAO Index Comparison\nClim vs Niño', fontsize=20)
ax.set_ylabel('Standardized NAO Index', fontsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
# Make x-axis labels bold
ax.set_xticklabels(labels, fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# Aggiustiamo i limiti dell'asse X per centrare la visualizzazione sui box ravvicinati
ax.set_xlim(0.5, 2.6)

# Salvataggio e visualizzazione
plt.savefig("out_img/nao/boxplot_comparison.png", dpi=300, bbox_inches='tight')
#plt.show()