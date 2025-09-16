import sys
import pandas as pd

# Usage: python check_correlation.py <csv_file> <col1> <col2>
# Example: python check_correlation.py detailed_pressure_data_noisy.csv Pressure Pressure_noisy

if len(sys.argv) < 4:
    print("Usage: python check_correlation.py <csv_file> <col1> <col2>")
    sys.exit(1)

csv_file = sys.argv[1]
col1 = sys.argv[2]
col2 = sys.argv[3]

df = pd.read_csv(csv_file)
if col1 not in df.columns or col2 not in df.columns:
    print(f"Columns '{col1}' and/or '{col2}' not found in {csv_file}")
    sys.exit(1)

corr = df[col1].corr(df[col2])
print(f"Pearson correlation between '{col1}' and '{col2}': {corr:.4f}") 