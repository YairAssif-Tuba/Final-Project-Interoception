import sys
import subprocess
import re
import importlib.util
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.py")

# Usage: python generate_all.py <trial_duration> <sampling_rate> <hrv_scale> [add_noise]
# Example: python generate_all.py 12 200 0.5 True

def update_config(trial_duration, sampling_rate, hrv_scale):
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if '"TRIAL_DURATION"' in line:
            line = re.sub(r':.*', f': {trial_duration},', line)
        elif '"SAMPLING_RATE"' in line:
            line = re.sub(r':.*', f': {sampling_rate},', line)
        elif '"HRV_LEVELS"' in line:
            line = re.sub(r':.*', f': [{hrv_scale}],   # stddev multiplier for RR intervals', line)
        new_lines.append(line)
    with open(CONFIG_PATH, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated config.py: TRIAL_DURATION={trial_duration}, SAMPLING_RATE={sampling_rate}, HRV_LEVELS=[{hrv_scale}]")

def get_config():
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {CONFIG_PATH}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_all.py <trial_duration> <sampling_rate> <hrv_scale> [add_noise]")
        sys.exit(1)
    trial_duration = int(sys.argv[1])
    sampling_rate = int(sys.argv[2])
    hrv_scale = float(sys.argv[3])
    add_noise_arg = None
    if len(sys.argv) > 4:
        add_noise_arg = sys.argv[4].lower() in ("true", "1", "yes")

    update_config(trial_duration, sampling_rate, hrv_scale)

    print("Running generate_synthetic_ecg.py...")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "generate_synthetic_ecg.py")], check=True)
    print("Running generate_pressure.py...")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "generate_pressure.py")], check=True)

    config = get_config()
    add_noise = config.get("ADD_PARABOLIC_NOISE", False)
    if add_noise_arg is not None:
        add_noise = add_noise_arg
        print(f"[Override] ADD_PARABOLIC_NOISE set to {add_noise} for this run.")
    else:
        print(f"ADD_PARABOLIC_NOISE from config.py: {add_noise}")

    if add_noise:
        print("Adding parabolic noise as specified in config.py...")
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "add_parabolic_noise.py")], check=True)
    else:
        print("No parabolic noise added.")
    print("Done.")

if __name__ == "__main__":
    main() 