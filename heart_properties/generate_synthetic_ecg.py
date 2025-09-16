"""
Synthetic ECG Signal Generator
-----------------------------
This script generates a synthetic electrocardiogram (ECG) signal using the neurokit2 library.
The generated signal is saved as a CSV file, which contains the following columns:
    - ECG: The simulated ECG voltage values (in millivolts) at each time point.
    - mean_hr: The mean heart rate (in beats per minute) used for this simulation.
    - hrv_scale: The heart rate variability (HRV) scale used for this simulation.

Parameters (from config.py):
    - TRIAL_DURATION: Length of the ECG signal in seconds, matching the trial duration.
    - SAMPLING_RATE: Number of samples per second (Hz).
    - OUTPUT_DIR: Directory where the CSV file will be saved.
    - MEAN_HR_RANGE: Range (min, max) of possible mean heart rates (bpm).
    - ENABLE_HRV: Whether to enable heart rate variability in the simulation.
    - HRV_LEVELS: List of possible HRV scale values (higher values = more variability between beats).

How it works:
    - A random mean heart rate is chosen from MEAN_HR_RANGE.
    - If HRV is enabled, a random HRV scale is chosen from HRV_LEVELS.
    - The ECG signal is simulated for the specified trial duration and sampling rate.
    - The signal and parameters are saved to a CSV file for further analysis or visualization.
"""
# heart_properties/generate_synthetic_ecg.py

import os
import pandas as pd
import neurokit2 as nk
import numpy as np
from config import config

# Load config
TRIAL_DURATION = config["TRIAL_DURATION"]
SAMPLING_RATE = config["SAMPLING_RATE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
HR_RANGE = config["MEAN_HR_RANGE"]
ENABLE_HRV = config["ENABLE_HRV"]
HRV_LEVELS = config["HRV_LEVELS"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_ecg(duration, sampling_rate, mean_hr, hrv_scale=0):
    if ENABLE_HRV:
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate,
                              heart_rate=mean_hr, noise=0.01)
    else:
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate,
                              heart_rate=mean_hr, noise=0.01)
    return ecg

# Generate single ECG signal
hr = np.random.uniform(*HR_RANGE)
hrv_scale = np.random.choice(HRV_LEVELS) if ENABLE_HRV else 0

ecg = simulate_ecg(TRIAL_DURATION, SAMPLING_RATE, hr, hrv_scale)

# Save to CSV
df = pd.DataFrame({
    "ECG": ecg,
    "mean_hr": hr,
    "hrv_scale": hrv_scale
})
df.to_csv(f"{OUTPUT_DIR}/high_hrv_ecg_signal.csv", index=False)

print(f"✅ Generated ECG signal with heart rate {hr:.1f} bpm and HRV scale {hrv_scale}")
