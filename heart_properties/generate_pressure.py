"""
Pressure Signal Generator
------------------------
This script converts synthetic ECG signals into pressure waveforms.
The generated signal contains the following columns:
    - ECG: The original ECG signal
    - Pressure: The derived pressure waveform in mmHg
    - R_Peaks: Binary markers for R-peak locations
    - Time: Time points in seconds
    - Phase_Bin: The phase bin number (1-10) for each time point
    - Normalized_Pressure: Pressure values normalized to 0-1 range

Parameters (from config.py):
    - TRIAL_DURATION: Length of the signal in seconds
    - SAMPLING_RATE: Number of samples per second (Hz)
    - OUTPUT_DIR: Directory where the CSV file will be saved
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from config import config

# Load config
TRIAL_DURATION = config["TRIAL_DURATION"]
SAMPLING_RATE = config["SAMPLING_RATE"]
OUTPUT_DIR = os.path.abspath(config["OUTPUT_DIR"])

# Blood pressure parameters
SYSTOLIC_RANGE = (110, 130)  # Normal systolic range in mmHg
DIASTOLIC_RANGE = (70, 85)   # Normal diastolic range in mmHg
PULSE_PRESSURE_VARIATION = 0.1  # 10% variation in pulse pressure

print(f"Output directory: {OUTPUT_DIR}")

def synthetic_pressure_waveform(length, systolic, diastolic):
    """
    Generates an arterial pressure pulse waveform of given length.
    Longer length = slower pressure decay (lower heart rate)
    
    Args:
        length: Number of samples in the pulse
        systolic: Systolic pressure in mmHg
        diastolic: Diastolic pressure in mmHg
    """
    t = np.linspace(0, 1, length)
    
    # Create pulse shape with main peak and dicrotic notch
    pulse = (
        1.5 * np.exp(-((t - 0.2)**2) / 0.002) +  # Main peak (systolic)
        0.5 * np.exp(-((t - 0.4)**2) / 0.01)     # Dicrotic notch
    )
    
    # Add natural pressure decay
    decay = np.exp(-t * 2)  # Exponential decay
    pulse = pulse * decay
    
    # Scale to mmHg range
    pulse = (pulse - pulse.min()) / (pulse.max() - pulse.min())  # Normalize to 0-1
    pulse = pulse * (systolic - diastolic) + diastolic  # Scale to mmHg range
    
    return pulse

def generate_pressure_from_ecg(ecg_signal, sampling_rate, mean_hr):
    """
    Generate pressure waveform from ECG signal with HRV-modulated pulse shapes.
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        mean_hr: Mean heart rate in bpm
    """
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    r_peaks = info["ECG_R_Peaks"]
    
    # Calculate base systolic and diastolic pressures
    base_systolic = np.mean(SYSTOLIC_RANGE)
    base_diastolic = np.mean(DIASTOLIC_RANGE)
    
    # Initialize pressure array with diastolic pressure
    pressure = np.ones_like(ecg_signal) * base_diastolic
    
    # Generate pressure for each heartbeat
    for i in range(len(r_peaks) - 1):
        r = r_peaks[i]
        next_r = r_peaks[i + 1]
        rr_interval = next_r - r
        
        # Add some variation to systolic pressure based on heart rate
        hr_variation = (mean_hr - 60) / 60  # Normalize to 0-1 range
        systolic = base_systolic * (1 + hr_variation * PULSE_PRESSURE_VARIATION)
        
        # Generate pressure pulse
        template = synthetic_pressure_waveform(
            length=rr_interval,
            systolic=systolic,
            diastolic=base_diastolic
        )
        
        # Add the pulse to the pressure signal
        end = min(len(pressure), r + rr_interval)
        pressure[r:end] = template[:end - r]
    
    return pressure, r_peaks

def slice_pressure_into_phase_bins(pressure_series, sampling_rate, bin_count=10, bin_duration_ms=100):
    """
    Converts a pressure waveform into a [T, K] array of phase bins.
    Each row is a time step; each column is a pressure phase bin.
    """
    bin_width = int((bin_duration_ms / 1000) * sampling_rate)
    num_frames = len(pressure_series) // bin_width
    pressure = pressure_series[:num_frames * bin_width]  # trim excess

    # Reshape into matrix of shape [T, K]
    reshaped = pressure.reshape(num_frames, bin_width)
    downsampled = np.stack([
        np.mean(reshaped[:, i::bin_count], axis=1) for i in range(bin_count)
    ], axis=1)

    return downsampled

def generate_pressure_data(ecg_signal, mean_hr, hrv_scale):
    """
    Generate pressure data from ECG signal and save to files.
    """
    # Generate pressure waveform
    pressure, r_peaks = generate_pressure_from_ecg(ecg_signal, SAMPLING_RATE, mean_hr)
    
    # Create time points
    time_points = np.arange(len(ecg_signal)) / SAMPLING_RATE
    
    # Generate phase bins
    pressure_bins = slice_pressure_into_phase_bins(pressure, SAMPLING_RATE)
    
    # Create detailed DataFrame
    df_detailed = pd.DataFrame({
        "Time": time_points,
        "ECG": ecg_signal,
        "Pressure": pressure,
        "R_Peaks": np.isin(np.arange(len(ecg_signal)), r_peaks).astype(int),
        "mean_hr": mean_hr,
        "hrv_scale": hrv_scale
    })
    
    # Add phase bin information
    bin_width = int((100 / 1000) * SAMPLING_RATE)  # 100ms bins
    phase_bins = np.repeat(np.arange(10), len(df_detailed) // 10 + 1)[:len(df_detailed)]
    df_detailed["Phase_Bin"] = phase_bins
    
    # Add normalized pressure (for visualization)
    df_detailed["Normalized_Pressure"] = (pressure - pressure.min()) / (pressure.max() - pressure.min())
    
    # Create original DataFrame
    df_original = pd.DataFrame({
        "ECG": ecg_signal,
        "Pressure": pressure,
        "R_Peaks": np.isin(np.arange(len(ecg_signal)), r_peaks).astype(int),
        "mean_hr": mean_hr,
        "hrv_scale": hrv_scale
    })
    
    # Save both CSV files
    detailed_output = os.path.join(OUTPUT_DIR, "detailed_pressure_data.csv")
    original_output = os.path.join(OUTPUT_DIR, "synthetic_signals.csv")
    
    print(f"Saving detailed data to: {detailed_output}")
    print(f"Saving original data to: {original_output}")
    
    df_detailed.to_csv(detailed_output, index=False)
    df_original.to_csv(original_output, index=False)
    
    # Save phase-binned pressure data
    bins_output = os.path.join(OUTPUT_DIR, "pressure_phase_bins.npy")
    print(f"Saving phase bins to: {bins_output}")
    np.save(bins_output, pressure_bins)
    
    print(f"✅ Generated detailed pressure data and saved to {detailed_output}")
    print(f"✅ Generated original signals and saved to {original_output}")
    print(f"✅ Saved pressure phase bins to {bins_output}")
    
    return df_detailed, pressure_bins

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    
    # Read the synthetic ECG data
    ecg_file = os.path.join(OUTPUT_DIR, "high_hrv_ecg_signal.csv")
    print(f"Looking for ECG file: {ecg_file}")
    
    if os.path.exists(ecg_file):
        df_ecg = pd.read_csv(ecg_file)
        ecg_signal = df_ecg['ECG'].values
        mean_hr = df_ecg['mean_hr'].iloc[0]
        hrv_scale = df_ecg['hrv_scale'].iloc[0]
        
        # Generate pressure data
        df, pressure_bins = generate_pressure_data(ecg_signal, mean_hr, hrv_scale)
    else:
        print(f"❌ ECG file not found: {ecg_file}")
        print("Please generate ECG signals first using generate_synthetic_ecg.py") 