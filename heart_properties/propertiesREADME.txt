# Heart Properties - Step by Step Guide

## Easiest: One-Command Data Generation
1. Use the all-in-one script to set your parameters and generate everything:
   ```
   python heart_properties/generate_all.py <trial_duration> <sampling_rate> <hrv_scale>
   ```
   - Example:
     ```
     python heart_properties/generate_all.py 12 200 0.5
     ```
   This will:
   - Update `config.py` with your chosen trial duration, sampling rate, and HRV scale
   - Run both `generate_synthetic_ecg.py` and `generate_pressure.py` for you
   - Output all the main data files (no plotting)

---

## TLDR
pip3 install neurokit2 pandas numpy matplotlib
pip3 install PyWavelets
python3 generate_synthetic_ecg.py
python3 generate_pressure.py

## Setup (only once)
1. Install the required packages:
   ```
   pip install neurokit2 pandas numpy matplotlib
   pip install PyWavelets
   ```

## Configuration
2. Open `config.py` and adjust these settings:
   - `TRIAL_DURATION`: How long your signal should be in seconds
   - `SAMPLING_RATE`: How many samples per second 
   - `OUTPUT_DIR`: Where to save your files
   - `MEAN_HR_RANGE`: Min and max heart rate (bpm)
   - `ENABLE_HRV`: Turn heart rate variability on/off
   - `HRV_LEVELS`: How much heart rate varies between beats

## Generate Data
3. Create an ECG signal:
   ```
   python heart_properties/generate_synthetic_ecg.py
   ```
   **Outcome**: Creates `high_hrv_ecg_signal.csv` with ECG data + heart rate info

4. Convert ECG to pressure:
   ```
   python heart_properties/generate_pressure.py
   ```
   **Outcome**: Creates 3 files:
   - `detailed_pressure_data.csv` (everything with time/phase info)
   - `synthetic_signals.csv` (just ECG + pressure + R-peaks)
   - `pressure_phase_bins.npy` (10-phase pressure matrix for neural networks)


## Visualize (optional)
5. Plot your signals:
   ```
   python heart_properties/plot_ecg.py
   python heart_properties/visualize_pressure.py
   ```
   **Outcome**: PNG files showing your ECG and pressure waveforms

## Further Optional: Add Parabolic Noise
6. Add parabolic, cyclic noise to any signal column:
   ```
   python heart_properties/add_parabolic_noise.py <input_csv> <column_name> <output_csv> [amplitude] [cycles]
   ```
   - Example (add noise to Pressure):
     ```
     python heart_properties/add_parabolic_noise.py ecg_data/generated_signals/detailed_pressure_data.csv Pressure ecg_data/generated_signals/detailed_pressure_data_noisy.csv 0.2 3
     ```
   **Outcome**: New CSV with a `_noisy` column added for the chosen signal.

   
## What you get
- **ECG signal**: Realistic heart electrical activity
- **Pressure signal**: Blood pressure in mmHg (110-130 systolic, 70-85 diastolic)
- **R-peaks**: Exact timing of heartbeats
- **Phase bins**: Pressure divided into 10 time phases for neural network input
- **Heart rate variability**: Natural variation in timing between beats