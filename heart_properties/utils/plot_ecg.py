import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ecg(df, col, title, color, alpha=1.0):
    time = [i/100 for i in range(len(df))]  # 100 Hz sampling rate
    plt.plot(time, df[col], color, label=title, alpha=alpha)

# Main logic
noisy_file = os.path.join("heart_properties", "ecg_data", "generated_signals", "detailed_pressure_data_noisy.csv")
if not os.path.exists(noisy_file):
    noisy_file = os.path.join("heart_properties", "ecg_data", "generated_signals", "detailed_pressure_data.csv")
    if not os.path.exists(noisy_file):
        print("No detailed_pressure_data_noisy.csv or detailed_pressure_data.csv found to plot.")
        exit(0)

df = pd.read_csv(noisy_file)

plt.figure(figsize=(15, 6))
plot_ecg(df, "ECG", "ECG (original)", 'b-', alpha=0.8)
if "ECG_noisy" in df.columns:
    plot_ecg(df, "ECG_noisy", "ECG (noisy)", 'r-', alpha=0.6)
plt.title("ECG and ECG_noisy from detailed_pressure_data_noisy.csv")
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend()

# Create output directory if it doesn't exist
plot_dir = os.path.join('heart_properties', 'plots')
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, 'ecg_noisy_comparison.png')

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved as '{plot_path}'")
plt.show() 