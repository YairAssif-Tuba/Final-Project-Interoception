import sys
import numpy as np
import pandas as pd
import os

try:
    from config import config
except ImportError:
    config = {}

# Usage: python add_parabolic_noise.py <input_csv> <column_name(s)> <output_csv> [amplitude] [cycles]
# Example: python add_parabolic_noise.py detailed_pressure_data.csv ECG Pressure noisy_pressure.csv 0.2 3
# If no arguments, use config.py settings

def add_parabolic_noise(signal, amplitude=0.1, cycles=2):
    n = len(signal)
    cycle_len = n // cycles
    noise = np.zeros(n)
    rng = np.random.default_rng()
    for i in range(cycles):
        start = i * cycle_len
        end = (i + 1) * cycle_len if i < cycles - 1 else n
        x = np.linspace(0, 1, end - start)
        phase = rng.uniform(-0.2, 0.2)
        x_shifted = (x + phase) % 1
        parabola = -4 * (x_shifted - 0.5) ** 2 + 1
        amp = amplitude * rng.uniform(0.5, 1.5)
        noise[start:end] = amp * parabola
    noise += rng.normal(0, amplitude * 0.05, n)
    return signal + noise

# np.tile explanation:
# np.tile(array, N) repeats the array N times in a row.
# Example: np.tile([1,2,3], 2) -> [1,2,3,1,2,3]

def main():
    # If no CLI args, use config
    if len(sys.argv) == 1:
        if not config.get("ADD_PARABOLIC_NOISE", False):
            print("Parabolic noise is disabled in config.py (ADD_PARABOLIC_NOISE=False). Exiting.")
            sys.exit(0)
        input_csv = os.path.join("ecg_data/generated_signals", "detailed_pressure_data.csv")
        columns = config.get("NOISE_COLUMN", ["Pressure"])
        if isinstance(columns, str):
            columns = [columns]
        output_csv = os.path.join("ecg_data/generated_signals", f"detailed_pressure_data_noisy.csv")
        amplitude = config.get("NOISE_AMPLITUDE", 0.2)
        cycles = config.get("NOISE_CYCLES", 3)
    elif len(sys.argv) < 5:
        print("Usage: python add_parabolic_noise.py <input_csv> <column_name(s)> <output_csv> [amplitude] [cycles]")
        print("Example: python add_parabolic_noise.py detailed_pressure_data.csv ECG Pressure noisy_pressure.csv 0.2 3")
        sys.exit(1)
    else:
        input_csv = sys.argv[1]
        # All args except first, last, and optional amplitude/cycles are columns
        output_csv = sys.argv[-3]
        amplitude = float(sys.argv[-2]) if len(sys.argv) > 5 else 0.1
        cycles = int(sys.argv[-1]) if len(sys.argv) > 6 else 2
        columns = sys.argv[2:-3]
        if not columns:
            print("No columns specified.")
            sys.exit(1)

    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    processed = []
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in {input_csv}, skipping.")
            continue
        df[column + '_noisy'] = add_parabolic_noise(df[column].values, amplitude, cycles)
        processed.append(column)
    if not processed:
        print("No valid columns to process. Exiting.")
        sys.exit(1)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved noisy signal(s) for {processed} to {output_csv}")

if __name__ == "__main__":
    main() 