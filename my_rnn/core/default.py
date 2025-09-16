"""Default configuration for interval timing tasks - SIMPLIFIED PIEZO VERSION

This module provides default hyperparameters and configuration for the
interval timing tasks with simplified piezo interface.

CHANGES:
- Removed pretraining-related parameters
- Simplified piezo configuration
- Clean parameter set for biologically realistic piezo interface
"""

import numpy as np
import task


def get_default_input_output_sizes(rule_name):
    """Get the input and output dimensions for a task.

    Args:
        rule_name: Name of the task rule

    Returns:
        tuple: (n_input, n_output) dimensions

    Raises:
        ValueError: If rule_name is not recognized
    """
    if rule_name == 'interval_production':
        return 2, 1
    elif rule_name == 'interval_comparison':
        return 2, 2
    elif rule_name == 'time_bisection':
        return 2, 2
    elif rule_name == 'gaussian_bisection':
        return 1, 2
    else:
        raise ValueError(f"Unknown rule: {rule_name}")


def get_default_hp(rule_name=None, random_seed=None, use_piezo=False):
    """Get default hyperparameters for model training.

    Args:
        rule_name: Name of the task rule (default: None)
        random_seed: Random seed for reproducibility (default: None)
        use_piezo: Whether to enable simplified piezo interface (default: False)

    Returns:
        hp: Dictionary of hyperparameters
    """
    # Default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(1000000)
    else:
        seed = random_seed

    # Default hyperparameters
    hp = {
        'rule_name': rule_name,

        # Network architecture
        'rnn_type': 'RNN',  # Type of RNN (RNN, LSTM, GRU)
        'n_rnn': 256,  # Number of recurrent units
        'activation': 'softplus',  # Activation function (relu, softplus, tanh)

        # Task dimensions
        'n_input': 2,  # Number of input units (will be overridden)
        'n_output': 2,  # Number of output units (will be overridden)

        # Training parameters
        'batch_size_train': 64,  # Batch size for training
        'batch_size_test': 1024,  # Batch size for testing
        'learning_rate': 0.0003,  # Learning rate
        'optimizer': 'adam',  # Optimizer (adam, sgd)

        # Regularization
        'l1_firing_rate': 0,  # L1 regularization on activity
        'l2_firing_rate': 0,  # L2 regularization on activity
        'l1_weight': 0,  # L1 regularization on weights
        'l2_weight': 0,  # L2 regularization on weights

        # RNN dynamics
        'tau': 20,  # Time constant (ms)
        'dt': 20,  # Discretization time step (ms)
        'alpha': 1,  # Discretization time step/time constant

        # Noise levels
        'sigma_rec': 0.05,  # Recurrent noise
        'sigma_x': 0.01,  # Input noise

        # Weight initialization
        'initial_std': 0.1,  # Initial standard deviation of recurrent weights

        # Random number generation
        'seed': seed,
        'rng': np.random.RandomState(seed),

        # Simplified piezo interface (disabled by default)
        'use_piezo': use_piezo,
    }

    # Add simplified piezo-specific parameters if enabled

    # Override input/output sizes based on rule name
    if rule_name is not None:
        n_input, n_output = get_default_input_output_sizes(rule_name)
        hp['n_input'] = n_input
        hp['n_output'] = n_output

    # Special configuration for time_bisection
    if rule_name == 'time_bisection':
        hp= {
            'rule_name': rule_name,
            'dt': 20,                      # Time step in ms
            'tau': 20,                    # Membrane time constant (ms)
            'n_input': 2,                  # One input channel (stimulus)
            'n_output': 2,                 # Two outputs: short vs long
            'n_rnn': 256,                   # Small network for CPU (try 16 or 32)
            'activation': 'softplus',      # Nonlinearity for hidden units
            'batch_size_train': 64,        # Smaller batch for CPU
            'batch_size_test': 512,        # Smaller test batch
            'alpha': 1,                 # Computed later as dt / tau
            'sigma_rec': 0.05,             # Recurrent noise std
            'sigma_x': 0.01,               # Input noise std
            'initial_std': 0.1,            # Initial recurrent weight std
            'l1_weight': 0.0,              # L1 regularization on weights (not used)
            'l2_weight': 0.0001,           # L2 regularization on weights (0.0001)
            'l1_firing_rate': 0.0,         # L1 reg on firing rate
            'l2_firing_rate': 0.0001,         # L2 reg on firing rate (0.0001)
            'optimizer': 'adam',           # Optimizer
            'learning_rate': 0.0002,        # Learning rate
            'loss_type': 'mse',            # Use mean squared error loss
            'target_cost_weight': 1.0,     # Weight for supervised target loss
            'use_separate_cost_mask': True, # Compute cost only on response period
            'seed' : seed,
            'rng': np.random.RandomState(seed)
        }
    elif rule_name == 'gaussian_bisection':
        hp= {
            'rule_name': rule_name,
            'dt': 20,                      # Time step in ms
            'tau': 20,                    # Membrane time constant (ms)
            'n_input': 1,                  # One input channel (stimulus)
            'n_output': 2,                 # Two outputs: short vs long
            'n_rnn': 256,                   # Small network for CPU (try 16 or 32)
            'activation': 'softplus',      # Nonlinearity for hidden units
            'batch_size_train': 64,        # Smaller batch for CPU
            'batch_size_test': 512,        # Smaller test batch
            'alpha': 1,                 # Computed later as dt / tau
            'sigma_rec': 0.05,             # Recurrent noise std
            'sigma_x': 0.01,               # Input noise std
            'initial_std': 0.1,            # Initial recurrent weight std
            'l1_weight': 0.0,              # L1 regularization on weights (not used)
            'l2_weight': 0.0001,           # L2 regularization on weights (0.0001)
            'l1_firing_rate': 0.0,         # L1 reg on firing rate
            'l2_firing_rate': 0.0001,         # L2 reg on firing rate (0.0001)
            'optimizer': 'adam',           # Optimizer
            'learning_rate': 0.0002,        # Learning rate
            'loss_type': 'mse',            # Use mean squared error loss
            'target_cost_weight': 1.0,     # Weight for supervised target loss
            'use_separate_cost_mask': True, # Compute cost only on response period
            'seed' : seed,
            'rng': np.random.RandomState(seed)
        }
    if use_piezo:
        hp.update({
            # Simplified piezo interface configuration
            'heartbeat_slice_size': 20,  # Size of cardiac pressure slices
            'piezo_connection_fraction': 0.15,  # Fraction of neurons connected to piezo

            # NOTE: No pretraining parameters needed!
            # NOTE: No complex signal processing parameters needed!
            # The simplified interface has no trainable parameters
        })

        print(f"✅ Simplified piezo interface enabled with {hp['piezo_connection_fraction'] * 100:.1f}% connectivity")
        print(f"🚫 No pretraining required - biologically realistic interface")
    else:
        print("🚫 Piezo interface disabled - using original network")

    return hp