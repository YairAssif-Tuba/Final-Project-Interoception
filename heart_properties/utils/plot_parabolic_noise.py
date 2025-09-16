import sys
import pandas as pd
import matplotlib.pyplot as plt

# Usage: python plot_parabolic_noise.py <csv_file> <original_column> <noisy_column>
# Example: python plot_parabolic_noise.py detailed_pressure_data_noisy.csv Pressure Pressure_noisy

def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_parabolic_noise.py <csv_file> <original_column> <noisy_column>")
        sys.exit(1)
    csv_file = sys.argv[1]
    orig_col = sys.argv[2]
    noisy_col = sys.argv[3]

    df = pd.read_csv(csv_file)
    if orig_col not in df.columns or noisy_col not in df.columns:
        print(f"Columns '{orig_col}' and/or '{noisy_col}' not found in {csv_file}")
        sys.exit(1)

    plt.figure(figsize=(12, 5))
    plt.plot(df[orig_col], label=f"Original ({orig_col})", linewidth=2)
    plt.plot(df[noisy_col], label=f"Noisy ({noisy_col})", alpha=0.7)
    plt.title(f"Comparison: {orig_col} vs {noisy_col}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 