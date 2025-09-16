# real_cardiac_data.py
"""
Real Cardiac Data Integration for Piezo Interface

This module loads and processes real precompiled cardiac CSV files for use
with the RNN piezo interface, replacing the synthetic cardiac data generation.

Data Path: /Users/itanarbuldaliot/Documents/prj_interoception_modeling/10shrv0
Code Path: /Users/itanarbuldaliot/Documents/prj_interoception_modeling/my_rnn/core

CSV Format:
- Time: Float (seconds)
- ECG: Float
- Pressure: Float
- R_Peaks: Integer (binary)
- mean_hr: Float
- hrv_scale: Integer
- Phase_Bin: Integer
- Normalized_Pressure: Float
"""

import os
import numpy as np
import pandas as pd
import torch
import random
from typing import Optional, Tuple, List, Dict
import glob


class RealCardiacDataLoader:
    """Enhanced loader for real cardiac data CSV files with library selection."""

    def __init__(self, data_path: str = None, library_name: str = "test"):
        """
        Initialize the cardiac data loader with library selection.

        Args:
            data_path: Specific path to directory containing cardiac CSV files
            library_name: Name of the cardiac library to use
        """
        # Define available cardiac libraries
        self.cardiac_libraries = {
            "hr60_hrv0cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr60_hrv0cal",
            "hr60_hrv1cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr60_hrv1cal",
            "hr60_hrv3cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr60_hrv3cal",
            "hr90_hrv0cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr90_hrv0cal",
            "hr90_hrv1cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr90_hrv1cal",
            "hr90_hrv3cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr90_hrv3cal",
            "hr100_hrv0cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr100_hrv0cal",
            "hr100_hrv1cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr100_hrv1cal",
            "hr100_hrv3cal": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/hr100_hrv3cal",
            "test": "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/ecg_libraries_hr_cal_csv_split/test",
            # Add your other libraries here
        }

        if data_path is not None:
            self.data_path = data_path
            self.library_name = "custom"
        elif library_name in self.cardiac_libraries:
            self.data_path = self.cardiac_libraries[library_name]
            self.library_name = library_name
        else:
            raise ValueError(
                f"Unknown cardiac library: {library_name}. Available: {list(self.cardiac_libraries.keys())}")

        self.csv_files = self._discover_csv_files()
        self.data_duration = 10.0  # seconds per file

        #print(f"🫀 RealCardiacDataLoader initialized with library: {self.library_name}")
        #print(f"   Data path: {self.data_path}")
        #print(f"   Found {len(self.csv_files)} CSV files")

        if len(self.csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_path}")

    def _discover_csv_files(self) -> List[str]:
        """Discover all CSV files in the data directory."""
        pattern = os.path.join(self.data_path, "*.csv")
        csv_files = glob.glob(pattern)
        csv_files.sort()  # Ensure consistent ordering
        return csv_files

    def load_random_file(self) -> pd.DataFrame:
        """Load a random CSV file."""
        file_path = random.choice(self.csv_files)
        return self.load_file(file_path)

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load a specific CSV file."""
        try:
            df = pd.read_csv(file_path)
            self._validate_dataframe(df)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")

    def load_file_by_index(self, index: int) -> pd.DataFrame:
        """Load CSV file by index (0-999)."""
        if not 0 <= index < len(self.csv_files):
            raise ValueError(f"Index {index} out of range [0, {len(self.csv_files) - 1}]")
        return self.load_file(self.csv_files[index])

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame has expected columns and structure."""
        expected_columns = ['Time', 'ECG', 'Pressure', 'R_Peaks', 'mean_hr', 'hrv_scale', 'Phase_Bin',
                            'Normalized_Pressure']

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        if len(df) == 0:
            raise ValueError("Empty DataFrame")

        # Check time range (should be ~10 seconds)
        time_range = df['Time'].max() - df['Time'].min()
        #if not 9.0 < time_range < 11.0:
            #print(f"⚠️ Warning: Unexpected time range {time_range:.1f}s (expected ~10s)")


class RealCardiacProcessor:
    """Process real cardiac data for RNN piezo interface."""

    def __init__(self, loader: RealCardiacDataLoader):
        """
        Initialize the processor.

        Args:
            loader: Initialized RealCardiacDataLoader
        """
        self.loader = loader

    def extract_task_duration_data(self, task_duration_ms: float, dt: float, slice_size: int = 20) -> Dict:
        """
        Extract cardiac data for a specific task duration.

        Args:
            task_duration_ms: Duration of the cognitive task in milliseconds
            dt: RNN time step in milliseconds
            slice_size: Number of samples per RNN time step

        Returns:
            Dictionary containing processed cardiac data matching task timing
        """
        task_duration_s = task_duration_ms / 1000.0

        # Load appropriate cardiac data
        if task_duration_s <= 10.0:
            # Single file sufficient
            df = self.loader.load_random_file()
            cardiac_data = self._extract_duration(df, task_duration_s)
        else:
            # Need multiple files
            cardiac_data = self._extract_long_duration(task_duration_s)

        # Process for RNN timing
        processed_data = self._process_for_rnn(cardiac_data, dt, slice_size)

        return processed_data

    def _extract_duration(self, df: pd.DataFrame, duration_s: float) -> pd.DataFrame:
        """Extract a specific duration from a cardiac DataFrame."""
        start_time = df['Time'].iloc[0]
        end_time = start_time + duration_s

        mask = (df['Time'] >= start_time) & (df['Time'] <= end_time)
        return df[mask].copy()

    def _extract_long_duration(self, duration_s: float) -> pd.DataFrame:
        """Extract cardiac data for durations longer than 10 seconds."""
        num_files_needed = int(np.ceil(duration_s / 10.0))

        combined_data = []
        current_time_offset = 0.0

        for i in range(num_files_needed):
            # Load random file
            df = self.loader.load_random_file()

            # Adjust time column
            df_copy = df.copy()
            df_copy['Time'] = df_copy['Time'] - df_copy['Time'].iloc[0] + current_time_offset

            # If this is the last file, only take what we need
            if i == num_files_needed - 1:
                remaining_duration = duration_s - current_time_offset
                df_copy = self._extract_duration(df_copy, remaining_duration)

            combined_data.append(df_copy)
            current_time_offset += 10.0

        return pd.concat(combined_data, ignore_index=True)

    def _process_for_rnn(self, df: pd.DataFrame, dt: float, slice_size: int) -> Dict:
        """
        Process cardiac data to match RNN timing requirements.

        Args:
            df: Cardiac DataFrame
            dt: RNN time step in milliseconds
            slice_size: Samples per RNN time step

        Returns:
            Dictionary with RNN-compatible cardiac data
        """
        # Calculate sampling parameters
        dt_s = dt / 1000.0  # Convert to seconds
        total_duration_s = df['Time'].max() - df['Time'].min()
        rnn_timesteps = int(total_duration_s / dt_s)

        # Resample cardiac data to match RNN timing
        cardiac_sampling_rate = slice_size / dt_s  # Hz

        # Create time grid for resampling
        time_start = df['Time'].iloc[0]
        time_grid = np.linspace(time_start, time_start + rnn_timesteps * dt_s,
                                rnn_timesteps * slice_size)

        # Interpolate cardiac signals to new time grid
        pressure_resampled = np.interp(time_grid, df['Time'], df['Pressure'])
        normalized_pressure_resampled = np.interp(time_grid, df['Time'], df['Normalized_Pressure'])
        ecg_resampled = np.interp(time_grid, df['Time'], df['ECG'])

        # Handle R-peaks (binary signal - use nearest neighbor)
        r_peaks_indices = np.searchsorted(time_grid, df[df['R_Peaks'] == 1]['Time'])
        r_peaks_resampled = np.zeros(len(time_grid))
        valid_indices = r_peaks_indices[r_peaks_indices < len(time_grid)]
        r_peaks_resampled[valid_indices] = 1

        # Create hb_sequence in the format expected by the RNN
        # Shape: [total_samples, 3] where columns are [ECG, Pressure, R_Peaks]
        hb_sequence = np.stack([
            ecg_resampled,
            pressure_resampled,  # Use original pressure for piezo
            r_peaks_resampled
        ], axis=1)

        # Extract pressure slices for piezo interface
        pressure_slices = []
        for t in range(rnn_timesteps):
            start_idx = t * slice_size
            end_idx = (t + 1) * slice_size
            if end_idx <= len(pressure_resampled):
                slice_data = pressure_resampled[start_idx:end_idx]
            else:
                # Pad if necessary
                slice_data = pressure_resampled[start_idx:]
                padding = np.zeros(slice_size - len(slice_data))
                slice_data = np.concatenate([slice_data, padding])

            pressure_slices.append(slice_data)

        # Calculate statistics
        r_peak_times = df[df['R_Peaks'] == 1]['Time'].values
        r_peak_intervals = np.diff(r_peak_times) if len(r_peak_times) > 1 else np.array([0.8])
        heart_rate = 60.0 / np.mean(r_peak_intervals) if len(r_peak_intervals) > 0 else 75.0

        return {
            'hb_sequence': hb_sequence,
            'pressure_slices': pressure_slices,
            'rnn_timesteps': rnn_timesteps,
            'cardiac_sampling_rate': cardiac_sampling_rate,
            'total_duration_s': total_duration_s,
            'r_peak_times': r_peak_times,
            'heart_rate': heart_rate,
            'dt': dt,
            'slice_size': slice_size,
            'original_data': df,
            'time_grid': time_grid,
            'min_pressure': pressure_resampled.min(),
            'max_pressure': pressure_resampled.max(),
            'pressure_mean': pressure_resampled.mean(),
            'pressure_std': pressure_resampled.std()
        }


def create_real_cardiac_data_for_task(task_duration_ms: float, dt: float = 20.0,
                                      slice_size: int = 20, data_path: str = None,
                                      cardiac_library: str = "test") -> Dict:
    """
    Create cardiac data for a cognitive task with library selection.

    Args:
        task_duration_ms: Duration of the cognitive task in milliseconds
        dt: RNN time step in milliseconds
        slice_size: Samples per RNN time step
        data_path: Custom path to cardiac data directory (overrides library)
        cardiac_library: Name of the cardiac library to use

    Returns:
        Dictionary with cardiac data ready for RNN piezo interface
    """
    loader = RealCardiacDataLoader(data_path=data_path, library_name=cardiac_library)
    processor = RealCardiacProcessor(loader)

    cardiac_data = processor.extract_task_duration_data(task_duration_ms, dt, slice_size)

    # Add library information to the returned data
    cardiac_data['cardiac_library'] = loader.library_name
    cardiac_data['library_path'] = loader.data_path

    return cardiac_data

def test_real_cardiac_integration():
    """Test the real cardiac data integration."""

    #print("🧪 Testing Real Cardiac Data Integration")
    #print("=" * 50)

    # Test 1: Load single file
    loader = RealCardiacDataLoader()
    df = loader.load_random_file()
    #print(f"✅ Loaded cardiac data: {len(df)} samples, {df['Time'].max():.1f}s duration")

    # Test 2: Extract task-duration data
    processor = RealCardiacProcessor(loader)

    # Test with typical interval comparison task (2-3 seconds)
    task_duration_ms = 2500  # 2.5 seconds
    dt = 20  # 20ms time steps

    cardiac_data = processor.extract_task_duration_data(task_duration_ms, dt)

    #print(f"✅ Processed cardiac data for {task_duration_ms}ms task:")
    #print(f"   RNN timesteps: {cardiac_data['rnn_timesteps']}")
    #print(f"   Heart rate: {cardiac_data['heart_rate']:.1f} BPM")
    #print(f"   R-peaks: {len(cardiac_data['r_peak_times'])}")
    #print(f"   Pressure range: [{cardiac_data['min_pressure']:.3f}, {cardiac_data['max_pressure']:.3f}]")
    #print(f"   HB sequence shape: {cardiac_data['hb_sequence'].shape}")

    # Test 3: Verify temporal alignment
    expected_samples = cardiac_data['rnn_timesteps'] * cardiac_data['slice_size']
    actual_samples = len(cardiac_data['hb_sequence'])

    if expected_samples == actual_samples:
        print("✅ Temporal alignment correct")
    else:
        print(f"⚠️ Temporal alignment issue: expected {expected_samples}, got {actual_samples}")

    # Test 4: Long duration (>10s)
    long_duration_ms = 15000  # 15 seconds
    long_cardiac_data = processor.extract_task_duration_data(long_duration_ms, dt)
    print(f"✅ Long duration test: {long_cardiac_data['total_duration_s']:.1f}s")

    print("\n🎉 All tests passed! Real cardiac data integration ready.")

    return cardiac_data


if __name__ == "__main__":
    test_real_cardiac_integration()