import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eofs.xarray import Eof
import os

def process_nao(filename, seasons):
    print(f"Processing {filename}...")
    
    # Extract name part for title
    name_part = filename[8:12]
    if name_part == 'nino':
        name_part = 'Niño'
    elif name_part == 'clim':
        name_part = 'Clim'
    title = filename[:3]+" - "+name_part
    
    # Configuration
    level = 2 # 500 hPa
    modo = 0
    
    # Load data
    try:
        z_djf = xr.open_dataset(filename)['gh']
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return

    z_djf = z_djf.sel(lat=slice(20, 90))

    # Compute anomalies by removing the time-mean
    z_djf = z_djf - z_djf.mean(dim='time')

    # Create an EOF solver
    coslat = np.cos(np.deg2rad(z_djf.coords['lat'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(z_djf, weights=wgts)

    # Retrieve the leading EOF
    # eof1 = solver.eofsAsCovariance(neofs=2) # Not used for index calculation directly in provided snippet logic
    
    # Calculate NAO index
    # We need the solver to project onto the pattern. 
    # In the original code, it projected onto eof1.isel(mode=modo, lev=level)
    # Replicating original logic:
    eof1 = solver.eofsAsCovariance(neofs=2)
    spatial_pattern = eof1.isel(mode=modo, lev=level)
    
    nao_i = []
    # Loop over time (slow but replicates original)
    for i in range(len(z_djf.coords['time'].values)):
        anomaly = z_djf.isel(time=i, lev=level)
        index_spatial = spatial_pattern * anomaly
        nao_i.append(index_spatial.sum())

    # Standardization
    nao_i = np.array(nao_i)
    dev_nao = np.std(nao_i)
    nao_i = nao_i / dev_nao
    mean_nao = np.mean(nao_i)
    dev_nao = np.std(nao_i) # Recalculate (should be 1)
    
    print(f"  mean {mean_nao}")
    print(f"  dev {dev_nao}")

    # FFT Analysis
    T = 3.1536e7          # Time interval (1 year in seconds)
    L = len(nao_i)        # Signal length
    print(f"  n samples {L}")
    
    fft_values = np.fft.fft(nao_i)
    freqs = np.fft.fftfreq(L, T)
    positive_freqs = freqs[:L//2]
    amplitudes = 2.0/L * np.abs(fft_values[:L//2])
    
    freqs_per_year = positive_freqs * (86400*365)
    periods_years = 1 / freqs_per_year
    
    # Define file suffix based on original logic (nino/clim)
    file_suffix = filename[8:12]

    # --- Plotting Functions ---
    
    # 1. Standard FFT Plot (Period on x-axis)
    plt.figure()
    plt.semilogx(periods_years, amplitudes)
    plt.title(f"FFT Analysis {title}")
    plt.xlabel("Period (years)")
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.40)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}", dpi=300)
    plt.close()

    # 2. Smoothed FFT Plot
    plt.figure()
    window_size = 5
    weights = np.ones(window_size) / window_size
    amplitudes_smooth = np.convolve(amplitudes, weights, mode='same')

    plt.semilogx(periods_years, amplitudes, alpha=0.5, label='Original')
    plt.semilogx(periods_years, amplitudes_smooth, 'r-', linewidth=2, label=f'Rolling Avg (w={window_size})')
    plt.title(f"FFT Analysis {title} (Smoothed)")
    plt.xlabel("Period (years)")
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.40)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}_smooth", dpi=300)
    plt.close()

    # 3. FFT Period 0-10
    plt.figure()
    plt.semilogx(periods_years, amplitudes)
    plt.title(f"FFT Analysis {title} (0-10 years)")
    plt.xlabel("Period (years)")
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.40)
    plt.xlim(right=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}_0_10", dpi=300)
    plt.close()

    # 4. FFT Period >10
    plt.figure()
    plt.semilogx(periods_years, amplitudes)
    plt.title(f"FFT Analysis {title} (>10 years)")
    plt.xlabel("Period (years)")
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.40)
    plt.xlim(left=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}_10_inf", dpi=300)
    plt.close()

    # 5. Frequency Domain Plot with Secondary Axis
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.plot(freqs_per_year, amplitudes)
    #ax.set_title(f"FFT Analysis {title}", fontsize=16)
    if filename == 'DJFmean_clim.nc':
        ax.set_title('Clim', fontsize=20, fontweight='bold')
    else:
        ax.set_title('Niño', fontsize=20, fontweight='bold')
    ax.set_xlabel("Frequency (cycles/year)", fontsize=16)
    ax.set_ylabel("Amplitude", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0, 0.40)
    ax.set_xlim(left=0.002)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    def freq2period(x):
        with np.errstate(divide='ignore'):
            return 1 / x

    def period2freq(x):
        with np.errstate(divide='ignore'):
            return 1 / x

    secax = ax.secondary_xaxis('top', functions=(freq2period, period2freq))
    secax.set_xlabel('Period (years)', fontsize=16)
    # Fix overlapping labels by setting manual ticks
    period_ticks = [2, 3, 4, 5, 6, 8, 10, 15, 25, 50]
    secax.set_ticks(period_ticks)
    secax.set_xticklabels(period_ticks)
    secax.tick_params(axis='x', labelsize=12)
    
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}_freq_domain", dpi=300)
    plt.close()

    # 6. Shuffled FFT for Verification
    nao_i_shuffled = nao_i.copy()
    np.random.shuffle(nao_i_shuffled)
    
    fft_values_shuffled = np.fft.fft(nao_i_shuffled)
    amplitudes_shuffled = 2.0/L * np.abs(fft_values_shuffled[:L//2])
    
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.plot(freqs_per_year, amplitudes_shuffled)
    
    if filename == 'DJFmean_clim.nc':
        ax.set_title('Clim (Shuffled)', fontsize=20, fontweight='bold')
    else:
        ax.set_title('Niño (Shuffled)', fontsize=20, fontweight='bold')
        
    ax.set_xlabel("Frequency (cycles/year)", fontsize=16)
    ax.set_ylabel("Amplitude", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0, 0.40)
    ax.set_xlim(left=0.002)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    secax = ax.secondary_xaxis('top', functions=(freq2period, period2freq))
    secax.set_xlabel('Period (years)', fontsize=16)
    secax.set_ticks(period_ticks)
    secax.set_xticklabels(period_ticks)
    secax.tick_params(axis='x', labelsize=12)
    
    plt.savefig(f"out_img/nao/{seasons}_FFT_{file_suffix}_shuffled", dpi=300)
    plt.close()
    
    print(f"Finished processing {filename}\n")

if __name__ == "__main__":
    files_to_process_DJF = ['DJFmean_clim.nc', 'DJFmean_nino.nc']
    files_to_process_JJA = ['JJAmean_clim.nc', 'JJAmean_nino.nc']
    # Create output directory if it doesn't exist
    os.makedirs("out_img/nao", exist_ok=True)
    
    for fname in files_to_process_DJF:
        process_nao(fname, 'DJF')
    
    for fname in files_to_process_JJA:
        process_nao(fname, 'JJA')