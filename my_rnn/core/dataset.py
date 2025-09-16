import torch
from torch.utils.data import Dataset, DataLoader
from real_cardiac_data import create_real_cardiac_data_for_task


# Change from absolute import to relative import
import task


class TaskDataset(Dataset):
    """Dataset class for training and validation, compatible with PyTorch DataLoader"""

    def __init__(self, rule_name, hp, mode='train', is_cuda=True, **kwargs):
        """Initialize the dataset with task parameters
        
        Args:
            rule_name (str): Name of the task rule to use
            hp (dict): Hyperparameters dictionary
            mode (str): 'train' or 'test'
            is_cuda (bool): Whether to use CUDA 
            **kwargs: Additional arguments passed to trial generation
        """
        self.rule_name = rule_name
        self.hp = hp
        self.is_cuda = is_cuda
        
        # Set batch size and task mode based on the operation mode
        if mode == 'train':
            self.batch_size = hp.get('batch_size_train', 64)
            self.task_mode = 'random'
        elif mode == 'test':
            self.batch_size = hp.get('batch_size_test', 64)
            self.task_mode = 'random_validate'
        else:
            raise ValueError(f'Unknown mode: {mode}')
        
        # Store additional parameters for trial generation
        self.kwargs = kwargs

    def __len__(self):
        """Return the dataset size (arbitrary large number for on-the-fly generation)"""
        return 1000000  # Arbitrary large number since data is generated on-the-fly

    def __getitem__(self, _):
        """Generate a batch of trials

        Args:
            _ (int): Batch index (unused as data is generated on-the-fly)

        Returns:
            dict: Dictionary containing the generated trials and associated data
        """
        # Generate trials using the task module
        self.trial = task.generate_trials(
            self.rule_name,
            self.hp,
            self.task_mode,
            batch_size=self.batch_size,
            **self.kwargs
        )
        #print(f"DEBUG: task.generate_trials returned")

        # Create result dictionary with necessary tensors
        result = {
            'inputs': torch.as_tensor(self.trial.x),
            'target_outputs': torch.as_tensor(self.trial.y),
            'cost_mask': torch.as_tensor(self.trial.cost_mask),
            'cost_start_time': 0,  # Default cost start time
            'cost_end_time': self.trial.x.shape[0],  # Use the sequence length
            'seq_mask': torch.ones(self.trial.x.shape[0], self.trial.x.shape[1]),
            'initial_state': torch.zeros((self.trial.x.shape[1], self.hp.get('n_rnn', 100)))
        }

        # Store task-specific parameters based on the rule name
        if hasattr(self.trial, 'epochs'):
            result['epochs'] = self.trial.epochs

        # Add task-specific parameters
        if self.rule_name == 'interval_production':
            if hasattr(self.trial, 'prod_interval'):
                result['prod_interval'] = self.trial.prod_interval
            if hasattr(self.trial, 'dly_interval'):
                result['dly_interval'] = self.trial.dly_interval

        elif self.rule_name == 'interval_comparison':
            if hasattr(self.trial, 'prod_interval1'):
                result['prod_interval1'] = self.trial.prod_interval1
            if hasattr(self.trial, 'prod_interval2'):
                result['prod_interval2'] = self.trial.prod_interval2
            if hasattr(self.trial, 'dly_interval'):
                result['dly_interval'] = self.trial.dly_interval


        elif self.rule_name == 'time_bisection' or self.rule_name == 'gaussian_bisection':

            if hasattr(self.trial, 'durations'):
                result['durations'] = self.trial.durations

            if hasattr(self.trial, 'is_long'):
                result['is_long'] = self.trial.is_long

            if hasattr(self.trial, 'short_standard'):
                result['short_standard'] = self.trial.short_standard

            if hasattr(self.trial, 'long_standard'):
                result['long_standard'] = self.trial.long_standard

            if hasattr(self.trial, 'comparison_intervals'):
                result['comparison_intervals'] = self.trial.comparison_intervals

        # MODIFIED: Real cardiac data integration for piezo and insula models
        if self.hp.get('use_piezo', False) or self.hp.get('use_insula', False):
            T = self.trial.x.shape[0]
            dt = self.hp.get('dt', 20)
            task_duration_ms = T * dt
            slice_size = self.hp.get('heartbeat_slice_size', 20)

            try:

                # Load real cardiac data for this task duration
                cardiac_data = create_real_cardiac_data_for_task(
                    task_duration_ms=task_duration_ms,
                    dt=dt,
                    slice_size=slice_size
                )

                # Use real cardiac data
                result['hb_sequence'] = torch.tensor(cardiac_data['hb_sequence'], dtype=torch.float32)

                # Add cardiac metadata for analysis
                result['cardiac_metadata'] = {
                    'heart_rate': cardiac_data['heart_rate'],
                    'r_peak_count': len(cardiac_data['r_peak_times']),
                    'total_duration_s': cardiac_data['total_duration_s'],
                    'pressure_stats': {
                        'min': cardiac_data['min_pressure'],
                        'max': cardiac_data['max_pressure'],
                        'mean': cardiac_data['pressure_mean'],
                        'std': cardiac_data['pressure_std']
                    },
                    'data_source': 'real_csv'
                }

                # Optional: Print debug info (remove this line for production)
                if hasattr(self, '_debug_cardiac') and self._debug_cardiac:
                    print(f"🫀 Loaded real cardiac data: {task_duration_ms}ms task, "
                          f"HR={cardiac_data['heart_rate']:.1f} BPM, "
                          f"{len(cardiac_data['r_peak_times'])} R-peaks")

            except Exception as e:
                print(f"⚠️ Failed to load real cardiac data: {e}")
                print(f"   Falling back to synthetic data generation")

                # Fallback to original synthetic generation (your existing code)
                cardiac_sampling_rate = slice_size / (dt / 1000)  # Hz
                task_duration_seconds = (T * dt) / 1000
                cardiac_samples_needed = int(task_duration_seconds * cardiac_sampling_rate)

                from cardiac_data import generate_simple_cardiac_pressure, create_sparse_hb_sequence

                # FIXED: Pass the calculated sampling rate instead of hardcoded 1000
                pressure = generate_simple_cardiac_pressure(
                    cardiac_samples_needed,
                    sampling_rate=cardiac_sampling_rate  # Dynamic!
                )
                hb_sequence, _ = create_sparse_hb_sequence(
                    pressure,
                    sampling_rate=cardiac_sampling_rate  # Dynamic!
                )

                result['hb_sequence'] = torch.tensor(hb_sequence, dtype=torch.float32)
                result['cardiac_metadata'] = {'data_source': 'synthetic_fallback'}

        return result


class TaskDatasetForRun:
    """Dataset class for running single trials (not for training)"""
    
    def __init__(self, rule_name, hp, noise_on=True, mode='test', **kwargs):
        """Initialize the dataset for running single trials
        
        Args:
            rule_name (str): Name of the task rule to use
            hp (dict): Hyperparameters dictionary
            noise_on (bool): Whether to add noise during trial generation
            mode (str): The running mode, typically 'test'
            **kwargs: Additional arguments passed to trial generation
        """
        self.rule_name = rule_name
        self.hp = hp
        self.noise_on = noise_on
        self.mode = mode
        self.kwargs = kwargs

    def __getitem__(self):
        """Generate a single trial
        
        Returns:
            dict: Dictionary containing the generated trial and associated data
        """
        # Generate a single trial using the task module
        self.trial = task.generate_trials(
            self.rule_name, 
            self.hp, 
            self.mode, 
            noise_on=self.noise_on, 
            batch_size=1,  # Single trial
            **self.kwargs
        )

        # Create result dictionary with necessary tensors
        result = {
            'inputs': torch.as_tensor(self.trial.x),
            'target_outputs': torch.as_tensor(self.trial.y),
            'cost_mask': torch.as_tensor(self.trial.cost_mask),
            'cost_start_time': 0,  # Default cost start time
            'cost_end_time': self.trial.x.shape[0],  # Use the sequence length
            'seq_mask': torch.ones(self.trial.x.shape[0], self.trial.x.shape[1]),
            'initial_state': torch.zeros((self.trial.x.shape[1], self.hp.get('n_rnn', 100)))
        }
        
        # Store task-specific parameters based on the rule name
        if hasattr(self.trial, 'epochs'):
            result['epochs'] = self.trial.epochs
            
        # Add task-specific parameters
        if self.rule_name == 'interval_production':
            if hasattr(self.trial, 'prod_interval'):
                result['prod_interval'] = self.trial.prod_interval
            if hasattr(self.trial, 'dly_interval'):
                result['dly_interval'] = self.trial.dly_interval
                
        elif self.rule_name == 'interval_comparison':
            if hasattr(self.trial, 'prod_interval1'):
                result['prod_interval1'] = self.trial.prod_interval1
            if hasattr(self.trial, 'prod_interval2'):
                result['prod_interval2'] = self.trial.prod_interval2
            if hasattr(self.trial, 'dly_interval'):
                result['dly_interval'] = self.trial.dly_interval


        elif self.rule_name == 'time_bisection' or self.rule_name == 'gaussian_bisection':

            if hasattr(self.trial, 'durations'):
                result['durations'] = self.trial.durations

            if hasattr(self.trial, 'p_long'):
                result['p_long'] = self.trial.p_long

            if hasattr(self.trial, 'short_standard'):
                result['short_standard'] = self.trial.short_standard

            if hasattr(self.trial, 'long_standard'):
                result['long_standard'] = self.trial.long_standard
        
        return result


class CardiacDataset(Dataset):
    """Dataset for cardiac pressure reconstruction pretraining - FIXED VERSION"""

    def __init__(self, hp, num_samples=100, T=50, slice_size=20, mode='train'):
        """Initialize cardiac dataset for piezo pretraining

        Args:
            hp (dict): Hyperparameters dictionary
            num_samples (int): Number of cardiac sequences to generate
            T (int): Number of timesteps per sequence
            slice_size (int): Number of samples per timestep (fixed at 20)
            mode (str): 'train' or 'test'
        """
        self.hp = hp
        self.num_samples = num_samples
        self.T = T
        self.slice_size = slice_size
        self.mode = mode

        # FIXED: Extract dt from hyperparameters
        dt = hp.get('dt', 20)  # Default to 20ms if not specified

        # Generate cardiac data once during initialization with dynamic dt
        from cardiac_data import generate_realistic_reconstruction_data

        total_samples = num_samples + (10 if mode == 'train' else 1)  # Extra for test

        # FIXED: Pass dt parameter to cardiac data generation
        sequences, targets, hb_sequences = generate_realistic_reconstruction_data(
            num_total=total_samples, T=T, slice_size=slice_size, dt=dt
        )

        if mode == 'train':
            self.sequences = torch.tensor(sequences[:num_samples], dtype=torch.float32)
            self.targets = torch.tensor(targets[:num_samples], dtype=torch.float32)
            self.hb_sequences = torch.tensor(hb_sequences[:num_samples], dtype=torch.float32)
        else:
            self.sequences = torch.tensor(sequences[-1:], dtype=torch.float32)
            self.targets = torch.tensor(targets[-1:], dtype=torch.float32)
            self.hb_sequences = torch.tensor(hb_sequences[-1:], dtype=torch.float32)

        print(f"CardiacDataset (FIXED): Generated {len(self.sequences)} {mode} sequences")
        print(f"  Using dt={dt}ms, slice_size={slice_size}")
        print(f"  Sampling rate: {slice_size / (dt / 1000):.1f} Hz")

    def __len__(self):
        """Return the number of cardiac sequences"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single cardiac sequence

        Args:
            idx (int): Index of the sequence to return

        Returns:
            dict: Dictionary containing cardiac data
        """
        if idx >= len(self.sequences):
            idx = idx % len(self.sequences)  # Handle wraparound for large datasets

        sequence = self.sequences[idx]
        target = self.targets[idx]
        hb_sequence = self.hb_sequences[idx]

        batch_size = 1  # Single sequence

        result = {
            'inputs': sequence.unsqueeze(0),  # Add batch dimension [1, T, slice_size]
            'target_outputs': target.unsqueeze(0),  # Add batch dimension [1, T, slice_size]
            'hb_sequence': hb_sequence.unsqueeze(0),  # Add batch dimension [1, T*slice_size, 3]
            'initial_state': torch.zeros((batch_size, self.hp.get('n_rnn', 100))),
            'mode': 'reconstruction'
        }

        return result


def create_cardiac_dataloader(hp, num_samples=100, batch_size=1, mode='train'):
    """Create a DataLoader for cardiac pretraining data - FIXED VERSION

    Args:
        hp (dict): Hyperparameters (now includes dt)
        num_samples (int): Number of cardiac sequences
        batch_size (int): Batch size (typically 1 for cardiac pretraining)
        mode (str): 'train' or 'test'

    Returns:
        DataLoader: PyTorch DataLoader for cardiac data
    """
    T = hp.get('pretraining_T', 50)
    slice_size = hp.get('heartbeat_slice_size', 20)

    # FIXED: CardiacDataset now uses dt from hp automatically
    dataset = CardiacDataset(hp, num_samples=num_samples, T=T, slice_size=slice_size, mode=mode)

    def cardiac_collate_fn(batch):
        """Custom collate function for cardiac data"""
        return batch[0]  # Return the single item (already has batch dimension)

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Always 1 for cardiac data
        shuffle=(mode == 'train'),
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=cardiac_collate_fn
    )

    return dataloader