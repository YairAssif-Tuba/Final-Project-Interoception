# my_rnn Core Module

This directory contains the core components of the neural network framework for interval timing tasks. The code is based on the models described in Bi and Zhou's paper on interval timing in recurrent neural networks.

## Directory Structure

- `core/`: The main module containing all core functionality
  - `tester/`: Test modules for different components
  - `model/`: Directory for storing trained models

## Key Components

### Task Generation
- `task.py`: Defines tasks (interval production, interval comparison, time bisection) and trial generation

### Network & Training
- `network.py`: Neural network architecture (RNN)
- `train.py`: Training functionality and model optimization
- `train_stepper.py`: Core training loop mechanics
- `cluster_training_unit.py`: Utility for parallel training on clusters - haven't been tested yet!

### Data & Inference
- `dataset.py`: Dataset classes for training and running models
- `run.py`: Interface for running trained models on tasks
- `default.py`: Default hyperparameters for each task
- `tools.py`: Utility functions

## Setup

### Dependencies
This codebase requires:
- PyTorch (>= 1.7.0)
- NumPy (>= 1.20.0)
- Matplotlib (for visualizations)
- Psignifit (for time bisection analysis)

You can install these with pip:
```bash
pip install torch numpy matplotlib psignifit
```

## How to Use

### Running a Trained Model

```python
from my_rnn.core.run import Runner

# Create a runner for a specific task
runner = Runner(
    rule_name='interval_production',  # Task type
    model_dir='path/to/model',        # Model directory
    is_cuda=True,                     # Use GPU if available
    noise_on=True                     # Include noise during inference
)

# Run the model on a specific trial
# Note: Required parameters depend on the task
trial, train_stepper = runner.run(
    prod_interval=600,   # Production interval in ms
    dly_interval=300     # Delay interval in ms
)

# Access model outputs
model_outputs = train_stepper.outputs.detach().cpu().numpy()
```

### Required Parameters for Each Task

#### Interval Production
- `prod_interval`: Duration to be reproduced (ms)
- `dly_interval`: Delay between stimulus and go cue (ms)

#### Interval Comparison
- `prod_interval1`: First interval duration (ms)
- `prod_interval2`: Second interval duration (ms)
- `dly_interval`: Delay between intervals (ms)

#### Time Bisection
- `short_standard`: Short reference duration (ms)
- `long_standard`: Long reference duration (ms)

### Running Tests

To run tests for a specific component:

```bash
cd my_rnn/core/tester
python dataset_tester.py      # Test dataset.py
python train_tester.py        # Test train.py
python run_tester.py --task interval_production  # Test run.py with specific task
```

To disable visualizations during testing:

```bash
python run_tester.py --task all --no-vis
```

## Training New Models

To train a new model:

```bash
python -m my_rnn.core.cluster_training_unit <rule_name> <weight_reg> <rate_reg> <model_index>
```

Example:
```bash
python -m my_rnn.core.cluster_training_unit interval_production 0.0001 0 0
```

This will train an interval_production model with:
- Weight L2 regularization = 0.0001
- Firing rate L2 regularization = 0
- Model index = 0

## References

This code implements models from:
- Bi, Z., & Zhou, C. (2020). "Understanding the computation of time using neural network models."
  *Proceedings of the National Academy of Sciences*, 117(19), 10530-10540. 