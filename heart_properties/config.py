# heart_properties/config.py

# Master config dict
config = {
    "N_SAMPLES": 1000,
    "DURATION": 10,
    "SAMPLING_RATE": 200,
    "OUTPUT_DIR": "ecg_data/generated_signals",

    # Heart rate control
    "MEAN_HR_RANGE": (60, 100),      # bpm

    # HRV control
    "ENABLE_HRV": True,
    "HRV_LEVELS": [0.5],   # stddev multiplier for RR intervals

    # Trial configuration
    "TRIAL_DURATION": 12,

    # Parabolic noise options
    "ADD_PARABOLIC_NOISE": False,      # Set to True to add noise automatically
    "NOISE_AMPLITUDE": 0.2,            # Amplitude of the parabolic noise
    "NOISE_CYCLES": 3,                 # Number of parabolic cycles
    "NOISE_COLUMN": "Pressure",       # Which column to add noise to
}
