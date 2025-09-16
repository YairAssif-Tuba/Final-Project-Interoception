import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

def plot_ecg(csv_file, title, subplot_pos):
    # Read the ECG data
    df = pd.read_csv(csv_file)
    
    # Create time points (in seconds)
    time = [i/100 for i in range(len(df))]  # 100 Hz sampling rate
    
    # Detect R peaks
    _, rpeaks = nk.ecg_peaks(df['ECG'], sampling_rate=100)
    
    # Create the subplot
    plt.subplot(subplot_pos)
    plt.plot(time, df['ECG'], 'b-', linewidth=1, label='ECG Signal')
    plt.plot(rpeaks['ECG_R_Peaks']/100, df['ECG'].iloc[rpeaks['ECG_R_Peaks']], 'ro', label='R Peaks')
    
    # Add RR intervals
    for i in range(len(rpeaks['ECG_R_Peaks'])-1):
        start = rpeaks['ECG_R_Peaks'][i]/100
        end = rpeaks['ECG_R_Peaks'][i+1]/100
        interval = end - start
        plt.text((start + end)/2, df['ECG'].iloc[rpeaks['ECG_R_Peaks'][i]] + 0.1, 
                f'{interval:.2f}s', ha='center', fontsize=8)
    
    plt.title(f'{title}\nHR: {df["mean_hr"].iloc[0]:.1f} bpm, HRV Scale: {df["hrv_scale"].iloc[0]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.legend()

# Create figure with two subplots
plt.figure(figsize=(15, 10))

# Plot both signals
plot_ecg("ecg_data/generated_signals/single_ecg_signal.csv", "Original ECG Signal", 211)
plot_ecg("ecg_data/generated_signals/high_hrv_ecg_signal.csv", "High HRV ECG Signal", 212)

# Adjust layout and save
plt.tight_layout()
plt.savefig('ecg_data/generated_signals/ecg_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'ecg_data/generated_signals/ecg_comparison.png'") 