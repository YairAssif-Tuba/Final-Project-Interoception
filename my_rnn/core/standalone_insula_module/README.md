# Standalone Insula Module

A completely self-contained insula RNN module for processing ECG signals and detecting heartbeat events. This module includes training capabilities and has no external code dependencies (except for PyTorch and standard libraries).

## Files

- `insula_module.py` - Main InsulaModule class for ECG processing
- `train.py` - Training script with validation-driven optimization
- `train_utils.py` - Self-contained utility functions (data loading, metrics, etc.)
- `example_usage.py` - Example usage with real ECG data
- `visualize_weights.py` - Weight visualization tools
- `config.json` - Configuration parameters and hyperparameters
- `README.md` - This file

## Architecture

The insula module contains 64 units organized as:
- **gINS**: 24 units (gustatory insula) - receives ECG input
- **dINS**: 24 units (dorsal insula) - intermediate processing  
- **aINS**: 16 units (anterior insula) - final processing and output

## Key Features

1. **Completely Standalone**: No external code dependencies beyond PyTorch
2. **Training Capabilities**: Full training pipeline with validation-driven optimization
3. **Heartbeat Detection**: Detects R-peaks in ECG signals using learned representations
4. **Flexible Usage**: Can be used for inference or training from scratch
5. **ECG Library Format**: Compatible with ecg_libraries_hr_cal_split/ format

## Usage

### Basic Inference

```python
from insula_module import InsulaModule
import numpy as np

# Initialize module (loads pretrained weights by default)
insula = InsulaModule(device='cuda')

# Process ECG signal
ecg_signal = np.array([...])  # ECG samples at 100Hz
aINS_activity = insula.process_ecg(ecg_signal)  # Returns [T, 16]

print(f"aINS activity shape: {aINS_activity.shape}")
```

### Training from Scratch

```python
# Initialize without pretrained weights
insula = InsulaModule(device='cuda', load_pretrained=False, freeze=False)

# Training is done via train.py script
```

### Training Script

```bash
python train.py \
  --train_ecg_library_path ../ecg_libraries_hr_cal_split/ecg_lib_train.npy \
  --val_ecg_library_path ../ecg_libraries_hr_cal_split/ecg_lib_val.npy \
  --steps 3000 \
  --batch_size 64 \
  --save_dir runs/my_training
```

### Example Usage

```bash
python example_usage.py \
  --ecg_library_path ../ecg_libraries_hr_cal_split/ecg_lib_test.npy \
  --checkpoint_dir runs/standalone_insula_train
```

## Configuration

Key parameters in `config.json`:

### Model Architecture
- **Total Units**: 64 insula units (24+24+16)
- **Activation**: tanh
- **Connectivity**: Fixed structural mask (gINS→gINS+dINS, dINS→dINS+aINS, aINS→aINS)

### Dynamics
- **Time Step**: 10ms
- **Tau**: 70ms  
- **Alpha**: 0.143 (leaky integration)
- **Noise**: 0.05 (sigma_rec)

### ECG Processing
- **Sampling Rate**: 100Hz
- **Z-score Epsilon**: 1e-8

### Peak Detection
- **Peak Threshold**: 0.7
- **Min Distance**: 300ms
- **Z-score Threshold**: 2.5
- **Refractory Period**: 300ms
- **Tolerance**: 50ms

### Training Hyperparameters
- **Focal Loss Alpha**: 0.239
- **Focal Loss Gamma**: 2.898
- **Beat Head Bias**: -3.0
- **Sparsity Weight**: 0.001

## Input Requirements

- **ECG Signal**: Numpy array from ecg_libraries format
- **Sampling Rate**: 100Hz (configurable)
- **Signal Length**: Any length (automatically z-scored per window)
- **Batch Processing**: Supports both single signals and batches

## Output

- **aINS Activity**: Tensor [T, 16] where T is the length of the ECG signal
- **16 aINS units**: Final insula processing units for heartbeat detection

## Training Features

1. **Validation-Driven**: Uses validation metrics to select best model
2. **Focal Loss**: Handles class imbalance in heartbeat detection
3. **Structural Constraints**: Maintains fixed connectivity patterns
4. **Comprehensive Metrics**: F1, precision, recall, IBI-MAE, dSDNN, dRMSSD
5. **Checkpointing**: Saves best and last checkpoints

## Dependencies

### Required
- PyTorch (>= 1.7.0)
- NumPy (>= 1.20.0)
- Matplotlib (for visualizations)
- SciPy (for signal processing)

### Optional
- CUDA (for GPU acceleration)

## Installation

```bash
pip install torch numpy matplotlib scipy
```

## Integration

This module can be easily integrated into larger systems:

```python
import torch.nn as nn
from insula_module import InsulaModule

class HeartbeatDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.insula = InsulaModule(load_pretrained=True)
        self.beat_head = nn.Linear(16, 1)  # 16 aINS units -> 1 output
    
    def forward(self, ecg_batch):
        aINS_activity = self.insula.process_ecg_batch(ecg_batch)  # [T, B, 16]
        logits = self.beat_head(aINS_activity)  # [T, B, 1]
        return logits.squeeze(-1)  # [T, B]
```

## Performance

The module is optimized for:
- **Speed**: Efficient forward pass with minimal overhead
- **Memory**: Compact 64-unit architecture
- **Accuracy**: Uses validation-driven training for optimal performance
- **Standalone**: No external code dependencies

## File Structure

```
standalone_insula_module/
├── insula_module.py      # Main module class
├── train.py             # Training script
├── train_utils.py       # Self-contained utilities
├── example_usage.py     # Usage examples
├── visualize_weights.py # Weight visualization
├── config.json          # Configuration
└── README.md           # This file
```