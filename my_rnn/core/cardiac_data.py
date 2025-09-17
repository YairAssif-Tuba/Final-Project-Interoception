"""
Cardiac Data Generation for Piezo Pretraining

This module provides functions to generate realistic cardiac pressure data
with sparse R-peaks for pretraining the PiezoInterface.
"""

import numpy as np
import torch


def generate_simple_cardiac_pressure(length, sampling_rate):
    """Generate realistic cardiac pressure with dynamic sampling rate

    Args:
        length: Number of samples to generate
        sampling_rate: Sampling rate in Hz (calculated dynamically from task dt)
    """

    # Generate R-peak timing first
    heart_rate = 75 + np.random.uniform(-12, 12)  # 63-87 BPM
    beat_interval_samples = int((60 / heart_rate) * sampling_rate)

    # Create time vector
    t = np.linspace(0, length / sampling_rate, length)
    pressure = np.zeros_like(t)

    # FIXED: Scale start delay with sampling rate
    current_pos = int(0.5 * sampling_rate)  # Start after 0.5 seconds
    r_peak_positions = []

    while current_pos < length:
        r_peak_positions.append(current_pos)

        # Define cardiac cycle phases (as fraction of beat interval)
        systolic_duration = int(0.3 * beat_interval_samples)  # 30% of cycle
        diastolic_duration = int(0.7 * beat_interval_samples)  # 70% of cycle

        # Generate systolic phase (rapid rise)
        systolic_end = min(current_pos + systolic_duration, length)
        if systolic_end > current_pos:
            systolic_t = np.linspace(0, np.pi, systolic_end - current_pos)
            systolic_pressure = np.sin(systolic_t) ** 2  # Sharp rise and fall
            pressure[current_pos:systolic_end] = systolic_pressure

        # Generate diastolic phase (gradual decline)
        diastolic_start = systolic_end
        diastolic_end = min(current_pos + beat_interval_samples, length)
        if diastolic_end > diastolic_start:
            diastolic_t = np.linspace(0, np.pi / 2, diastolic_end - diastolic_start)
            diastolic_pressure = np.cos(diastolic_t) ** 2  # Gradual decline
            pressure[diastolic_start:diastolic_end] = diastolic_pressure * 0.4  # Lower amplitude

        # Move to next beat with heart rate variability
        variability = int(0.1 * beat_interval_samples)
        next_interval = beat_interval_samples + np.random.randint(-variability, variability + 1)
        current_pos += next_interval

    # Add baseline and small amount of noise
    pressure += 1.0  # Baseline pressure
    pressure += 0.05 * np.random.randn(len(pressure))  # Small noise

    # Scale to reasonable range
    pressure = pressure * 1.5 + 0.5

    return pressure


def create_sparse_hb_sequence(pressure_vector, sampling_rate):
    """Create heartbeat sequence with R-peaks properly aligned before systolic peaks

    Args:
        pressure_vector: Cardiac pressure signal
        sampling_rate: Sampling rate in Hz (dynamic)
    """
    ecg_col = np.zeros_like(pressure_vector)
    rpeak_col = np.zeros_like(pressure_vector)

    # Find systolic peaks (local maxima)
    r_peak_positions = []

    # FIXED: Scale minimum distance with sampling rate
    min_distance = int(0.6 * sampling_rate)  # Minimum 0.6 seconds between peaks
    min_height = np.mean(pressure_vector) + 0.5 * np.std(pressure_vector)  # Threshold for peaks

    # Simple peak detection
    for i in range(min_distance, len(pressure_vector) - min_distance):
        # Check if this is a local maximum above threshold
        is_peak = True
        current_value = pressure_vector[i]

        # Must be above threshold
        if current_value < min_height:
            continue

        # Check if it's higher than neighbors in a window
        # FIXED: Scale window size with sampling rate
        window_size = int(0.05 * sampling_rate)  # 50ms window
        for j in range(max(0, i - window_size), min(len(pressure_vector), i + window_size + 1)):
            if pressure_vector[j] > current_value:
                is_peak = False
                break

        if is_peak:
            # Check minimum distance from previous R-peak
            if not r_peak_positions or (i - r_peak_positions[-1]) >= min_distance:
                # Place R-peak slightly BEFORE the pressure peak (electrical precedes mechanical)
                # FIXED: Scale R-peak offset with sampling rate
                r_peak_offset = int(0.02 * sampling_rate)  # 20ms before pressure peak
                r_peak_position = max(0, i - r_peak_offset)

                rpeak_col[r_peak_position] = 1
                r_peak_positions.append(r_peak_position)

    return np.stack([ecg_col, pressure_vector, rpeak_col], axis=1), r_peak_positions


def generate_realistic_reconstruction_data(num_total=101, T=50, slice_size=20, dt=20):
    """Generate REALISTIC cardiac pressure data with dynamic sampling rate

    Args:
        num_total: Number of sequences to generate
        T: Number of time steps per sequence
        slice_size: Number of samples per time step (fixed at 20)
        dt: Time step duration in ms (configurable)
    """

    # FIXED: Calculate dynamic sampling rate
    sampling_rate = slice_size / (dt / 1000)  # Hz

    sequences, targets, hb_sequences = [], [], []

    for i in range(num_total):
        # Generate REALISTIC cardiac pressure signal with dynamic sampling rate
        pressure = generate_simple_cardiac_pressure(T * slice_size, sampling_rate)

        # Input: pressure slices [T, slice_size]
        sequence = pressure.reshape(T, slice_size)

        # Target: SAME pressure slices (reconstruction task)
        target = sequence.copy()  # [T, slice_size] - reconstruct the input

        sequences.append(sequence)
        targets.append(target)

        # Create SPARSE heartbeat sequence with dynamic sampling rate
        hb_sequence, r_peaks = create_sparse_hb_sequence(pressure, sampling_rate)
        hb_sequences.append(hb_sequence)

        if i == 0:  # Debug first sequence
            print(f"Cardiac data generation (FIXED):")
            print(f"  Task dt: {dt}ms")
            print(f"  Slice size: {slice_size} samples")
            print(f"  Sampling rate: {sampling_rate:.1f} Hz")
            print(f"  Pressure range: {pressure.min():.3f} to {pressure.max():.3f}")
            print(f"  Sequence shape: {sequence.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  R-peaks: {len(r_peaks)} at positions {r_peaks}")
            print(f"  Time per slice: {slice_size / sampling_rate * 1000:.1f}ms")

    return (np.stack(sequences), np.stack(targets), np.stack(hb_sequences))


def prepare_cardiac_training_data(num_train=100, num_test=1, T=50, slice_size=20, dt=20):
    """
    Prepare cardiac training data for piezo pretraining with dynamic sampling rate

    Args:
        num_train: Number of training sequences
        num_test: Number of test sequences
        T: Number of time steps per sequence
        slice_size: Number of samples per time step (fixed at 20)
        dt: Time step duration in ms (configurable)

    Returns:
        dict: Contains train/test data as torch tensors
    """
    print("=== GENERATING CARDIAC PRETRAINING DATA (FIXED) ===")

    # Generate data with dynamic sampling rate
    sequences, targets, hb_sequences = generate_realistic_reconstruction_data(
        num_total=num_train + num_test, T=T, slice_size=slice_size, dt=dt
    )

    # Convert to tensors
    data = {
        'train_inputs': torch.tensor(sequences[:num_train], dtype=torch.float32),
        'train_targets': torch.tensor(targets[:num_train], dtype=torch.float32),
        'train_hbseqs': torch.tensor(hb_sequences[:num_train], dtype=torch.float32),
        'test_input': torch.tensor(sequences[-1], dtype=torch.float32),
        'test_target': torch.tensor(targets[-1], dtype=torch.float32),
        'test_hbseq': torch.tensor(hb_sequences[-1], dtype=torch.float32)
    }

    print(f"Generated {num_train} training sequences and {num_test} test sequence")
    print(f"Each sequence: {T} timesteps × {slice_size} samples per timestep")
    print(f"Time per timestep: {slice_size / (slice_size / (dt / 1000)):.1f}ms")

    return data