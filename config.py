"""
Configuration file for the Interoception Modeling project.

This file contains all configurable paths and settings to make the codebase
portable across different systems and users.

To use this configuration:
1. Update the paths below to match your system
2. Import this module in your scripts: from config import PATHS, SETTINGS
3. Use PATHS['key'] instead of hardcoded paths
"""

import os
from pathlib import Path

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# =============================================================================
# PATH CONFIGURATIONS
# =============================================================================

PATHS = {
    # Project structure paths
    "PROJECT_ROOT": PROJECT_ROOT,
    "MY_RNN_ROOT": PROJECT_ROOT / "my_rnn",
    "CORE_DIR": PROJECT_ROOT / "my_rnn" / "core",
    "TESTER_DIR": PROJECT_ROOT / "my_rnn" / "tester",
    "TRAINER_DIR": PROJECT_ROOT / "my_rnn" / "trainer",
    "HEART_PROPERTIES_DIR": PROJECT_ROOT / "heart_properties",
    
    # Model and data directories
    "MODEL_BASE_DIR": PROJECT_ROOT / "model",
    "CLUSTER_MODELS_DIR": PROJECT_ROOT / "model" / "cluster_training",
    "RESULTS_BASE_DIR": PROJECT_ROOT / "results",
    "DATA_BASE_DIR": PROJECT_ROOT / "data",
    "LOGS_BASE_DIR": PROJECT_ROOT / "logs",
    
    # Cardiac data extracted from zip file - UPDATE THIS FOR YOUR SYSTEM
    "CARDIAC_DATA_EXTRACTED": PROJECT_ROOT / "data" / "HB_dataset",
    
    # Paths within the extracted cardiac data
    "CARDIAC_DATA_BASE": PROJECT_ROOT / "data" / "HB_dataset", 
    "ECG_SPLIT_DIR": PROJECT_ROOT / "data" / "HB_dataset" / "ecg_libraries_hr_cal_csv_split",

    # Analysis results paths
    "LIBRARY_RESULTS_BASE": PROJECT_ROOT / "library_results",
    "TASK_TESTER_RESULTS_BASE": PROJECT_ROOT / "task_tester_results",
    
    # Dataset paths
    "ENHANCED_DATASETS_DIR": PROJECT_ROOT / "enhanced_interval_datasets",
}

# Cardiac library configurations - specific heart rate/HRV combinations
CARDIAC_LIBRARIES = {
    "hr60_hrv0cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr60_hrv0cal"),
    "hr60_hrv1cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr60_hrv1cal"), 
    "hr60_hrv3cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr60_hrv3cal"),
    "hr90_hrv0cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr90_hrv0cal"),
    "hr90_hrv1cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr90_hrv1cal"),
    "hr90_hrv3cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr90_hrv3cal"), 
    "hr100_hrv0cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr100_hrv0cal"),
    "hr100_hrv1cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr100_hrv1cal"),
    "hr100_hrv3cal": str(PATHS["CARDIAC_DATA_BASE"] / "hr100_hrv3cal"),
}

# ECG library files for insula module training/testing
ECG_LIBRARIES = {
    "train": str(PATHS["ECG_SPLIT_DIR"] / "ecg_lib_train.npy"),
    "val": str(PATHS["ECG_SPLIT_DIR"] / "ecg_lib_val.npy"), 
    "test": str(PATHS["ECG_SPLIT_DIR"] / "ecg_lib_test.npy"),
}

# Analysis results directories
ANALYSIS_RESULTS = {
    "time_bisection_piezo": PATHS["LIBRARY_RESULTS_BASE"] / "time_bisection_piezo",
    "time_bisection_insula": PATHS["LIBRARY_RESULTS_BASE"] / "time_bisection_insula",
    "interval_comparison_piezo": PATHS["LIBRARY_RESULTS_BASE"] / "interval_comparison_piezo",
    "interval_comparison_insula": PATHS["LIBRARY_RESULTS_BASE"] / "interval_comparison_insula",
    "interval_production_piezo": PATHS["LIBRARY_RESULTS_BASE"] / "interval_production_piezo", 
    "interval_production_insula": PATHS["LIBRARY_RESULTS_BASE"] / "interval_production_insula",
}

# Task tester results paths
TESTER_RESULTS = {
    "time_bisection_piezo": PATHS["TASK_TESTER_RESULTS_BASE"] / "time_bisection_piezo",
    "time_bisection_insula": PATHS["TASK_TESTER_RESULTS_BASE"] / "time_bisection_insula",
    "interval_comparison_piezo": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_comparison_piezo",
    "interval_comparison_insula": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_comparison_insula",
    "interval_production_piezo": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_production_piezo",
    "interval_production_insula": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_production_insula",
    "time_bisection_standard": PATHS["TASK_TESTER_RESULTS_BASE"] / "time_bisection_standard",
    "interval_comparison_standard": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_comparison_standard",
    "interval_production_standard": PATHS["TASK_TESTER_RESULTS_BASE"] / "interval_production_standard",
}
# =============================================================================
# GENERAL SETTINGS
# =============================================================================

SETTINGS = {
    # Model training settings
    "DEFAULT_CUDA": True,
    "DEFAULT_NOISE_ON": True,
    "DEFAULT_SEED": 42,
    
    # Analysis settings
    "DEFAULT_N_TRIALS": 100,
    "DEFAULT_BATCH_SIZE": 1,
    
    # Visualization settings
    "DEFAULT_DPI": 300,
    "DEFAULT_FIGSIZE": (10, 6),
    "SAVE_PLOTS": True,
    
    # File formats
    "MODEL_EXTENSION": ".pth",
    "RESULTS_EXTENSION": ".json",
    "PLOT_EXTENSION": ".png",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_model_path(rule_name, model_params="", model_index=0):
    """Generate a standardized model path."""
    if model_params:
        model_dir = PATHS["MODEL_BASE_DIR"] / rule_name / model_params / f"model_{model_index}"
    else:
        model_dir = PATHS["MODEL_BASE_DIR"] / rule_name / f"model_{model_index}"
    return ensure_dir(model_dir)

def get_cluster_model_path(rule_name, w2_reg, r2_reg, model_index=0):
    """Generate a standardized cluster training model path."""
    model_params = f'w2_{w2_reg}_r2_{r2_reg}'
    model_dir = PATHS["CLUSTER_MODELS_DIR"] / rule_name / model_params / f"model_{model_index}"
    return ensure_dir(model_dir)

def get_results_path(analysis_type, filename):
    """Generate a standardized results path."""
    results_dir = ANALYSIS_RESULTS.get(analysis_type, PATHS["RESULTS_BASE_DIR"])
    ensure_dir(results_dir)
    return results_dir / filename

def get_tester_results_path(tester_type, filename):
    """Generate a standardized tester results path."""
    results_dir = TESTER_RESULTS.get(tester_type, PATHS["RESULTS_BASE_DIR"])
    ensure_dir(results_dir)
    return results_dir / filename

def get_cardiac_library_path(library_name):
    """Get the path for a specific cardiac library."""
    return CARDIAC_LIBRARIES.get(library_name)

def get_ecg_library_path(split_name):
    """Get the path for ECG library files (train/val/test)."""
    return ECG_LIBRARIES.get(split_name)

def get_dataset_path(task_name):
    """Get the path for enhanced dataset files."""
    dataset_files = {
        'interval_comparison': 'interval_comparison_dataset.json',
        'interval_production': 'interval_production_dataset.json', 
        'time_bisection': 'time_bisection_dataset.json'
    }
    if task_name in dataset_files:
        return PATHS["ENHANCED_DATASETS_DIR"] / dataset_files[task_name]
    return None

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that critical paths exist."""
    critical_paths = [
        PATHS["CARDIAC_DATA_BASE"],
        PATHS["ECG_SPLIT_DIR"],
    ]
    
    missing_paths = []
    for path in critical_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("⚠️  WARNING: The following critical paths do not exist:")
        for path in missing_paths:
            print(f"   - {path}")
        print("\n📝 Expected cardiac data structure:")
        print("   data/cardiac_data/")
        print("   ├── hr60_hrv0cal/")
        print("   ├── hr60_hrv1cal/")  
        print("   ├── hr60_hrv3cal/")
        print("   ├── hr90_hrv0cal/")
        print("   ├── hr90_hrv1cal/")
        print("   ├── hr90_hrv3cal/")
        print("   ├── hr100_hrv0cal/")
        print("   ├── hr100_hrv1cal/")
        print("   ├── hr100_hrv3cal/")
        print("   └── ecg_libraries_hr_cal_split/")
        print("       ├── ecg_lib_train.npy")
        print("       ├── ecg_lib_val.npy")
        print("       └── ecg_lib_test.npy")
        print("\n💡 Please extract your cardiac data zip to: data/cardiac_data/")
        print("   See the comments in config.py for guidance.")
    else:
        print("✅ All critical paths validated successfully.")
    
    return len(missing_paths) == 0

# Auto-validate when imported (can be disabled by setting SKIP_VALIDATION=True)
if not os.environ.get('SKIP_VALIDATION'):
    validate_config()
