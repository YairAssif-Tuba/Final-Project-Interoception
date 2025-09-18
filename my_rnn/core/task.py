"""Collections of tasks focused on Interval Production and Interval Comparison."""

from __future__ import division
import random
import numpy as np
import math
import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..'))
from config import get_dataset_path


class DatasetLoader:
    def __init__(self, dataset_path):
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
            self.train_data = self.dataset['train']
            self.index = 0
            self.total_samples_used = 0
            print(f"Loaded dataset with {len(self.train_data)} training samples")
        else:
            self.train_data = None

    def get_next_params(self, batch_size):
        if self.train_data is None:
            return None
            # ADD THIS DEBUG (only print first few times):
        if self.total_samples_used < 5:
            print(f"DEBUG: DatasetLoader providing {batch_size} samples from built-in task module")

        params = []
        for _ in range(batch_size):
            params.append(self.train_data[self.index % len(self.train_data)])
            self.index += 1
            self.total_samples_used += 1

        if self.total_samples_used % 1000 == 0:
            print(f"DATASET VERIFICATION: Used {self.total_samples_used} samples")
        return params


# Global dataset loaders
_dataset_loaders = {}


def get_dataset_loader(task_name):
    if task_name not in _dataset_loaders:
        dataset_path = get_dataset_path(task_name)
        if dataset_path and os.path.exists(dataset_path):
            _dataset_loaders[task_name] = DatasetLoader(str(dataset_path))
        else:
            _dataset_loaders[task_name] = None
    return _dataset_loaders[task_name]
# ADD THIS DEBUG PRINT
#print(f"DEBUG: task.py module being imported/reloaded at {id(__name__)}")
#mport traceback
#traceback.print_stack(limit=10)

class Trial(object):
    """Class representing a batch of trials for timing tasks."""

    def __init__(self, config, xtdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            xtdim: int, number of total time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.xtdim = xtdim

        # time major arrays
        self.x = np.zeros((xtdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        self.cost_mask = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        
        # strength of input noise
        self._sigma_x = config['sigma_x'] * math.sqrt(2./self.config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths):
        """Add an input or stimulus output to the indicated channel.

        Args:
            loc_type: str type of information ('input', 'out', or 'cost_mask')
            loc_idx: index of channel
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
        """

        if loc_type == 'input':
            # Add to input channels
            for i in range(self.batch_size):
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'out':
            # Add to output targets
            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'cost_mask':
            # Add to cost mask
            for i in range(self.batch_size):
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]

        else:
            raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x
        
    def visualize(self, trial_idx=0, save_path=None, title=None):
        """Visualize a single trial from the batch.
        
        Args:
            trial_idx: Index of the trial to visualize
            save_path: Path to save the figure (if None, just displays it)
            title: Optional title for the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for visualization")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Plot input channels
        ax = axes[0]
        for i in range(self.n_input):
            ax.plot(self.x[:, trial_idx, i], label=f'Input {i}')
        ax.set_ylabel('Input')
        ax.legend()
        
        # Plot output channels
        ax = axes[1]
        for i in range(self.n_output):
            ax.plot(self.y[:, trial_idx, i], label=f'Output {i}')
        ax.set_ylabel('Target Output')
        ax.legend()
        
        # Plot cost mask
        ax = axes[2]
        for i in range(self.n_output):
            ax.plot(self.cost_mask[:, trial_idx, i], label=f'Mask {i}')
        ax.set_ylabel('Cost Mask')
        ax.set_xlabel('Time Steps')
        ax.legend()
        
        # Add title if provided
        if title is not None:
            plt.suptitle(title)
            
        # Add epoch boundaries if available
        if hasattr(self, 'epochs'):
            for ax in axes:
                for name, (start, end) in self.epochs.items():
                    if start is not None:
                        ax.axvline(x=start[trial_idx], color='gray', linestyle='--', alpha=0.5)
                    if end is not None:
                        ax.axvline(x=end[trial_idx], color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def _interval_production(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)

    :param mode: the mode of generating. Options: 'random', 'random_validate', 'test'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param prod_interval, dly_interval: used only for 'test' mode

    :return: Trial object
    '''
    dt = config['dt']
    rng = config.get('rng', np.random.RandomState())
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # TRY DATASET FIRST - ADD THIS BLOCK:
        dataset_loader = get_dataset_loader('interval_production')
        if dataset_loader and dataset_loader.train_data is not None:
            params = dataset_loader.get_next_params(batch_size)
            prod_interval = (np.array([p['prod_interval'] for p in params]) / dt).astype(int)
            dly_interval = (np.array([p['dly_interval'] for p in params]) / dt).astype(int)
        else:
            # FALLBACK TO ORIGINAL CODE:
            #prod_interval = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
            #dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)
            # Fallback to enhanced intervals instead of short ones
            prod_interval = (rng.uniform(1200, 2400, batch_size) / dt).astype(int)
            dly_interval = (rng.uniform(1000, 1400, batch_size) / dt).astype(int)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        #prod_interval = (rng.uniform(600, 1200, batch_size)/dt).astype(int)
        #dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)
        # Use same enhanced intervals as training
        dataset_loader = get_dataset_loader('interval_production')
        if dataset_loader and dataset_loader.train_data is not None:
            params = dataset_loader.get_next_params(batch_size)
            prod_interval = (np.array([p['prod_interval'] for p in params]) / dt).astype(int)
            dly_interval = (np.array([p['dly_interval'] for p in params]) / dt).astype(int)
        else:
            # Fallback to enhanced intervals instead of short ones
            prod_interval = (rng.uniform(1200, 2400, batch_size) / dt).astype(int)
            dly_interval = (rng.uniform(1000, 1400, batch_size) / dt).astype(int)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)

        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration

    # the onset time of the second stimulus
    stim2_on = stim1_off + prod_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration

    # the onset time of the go cue
    control_on = stim2_off + dly_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration

    # response start time
    response_on = control_off + prod_interval
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # pulse input
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))

    # go cue
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))

    # output
    trial.add('out', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'delay': (stim2_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.dly_interval = dly_interval
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def interval_production(config, mode, **kwargs):
    '''Wrapper function for interval production task'''
    return _interval_production(config, mode, **kwargs)


def _interval_comparison(config, mode, **kwargs):
    '''
    Two stimuli are successively presented, the network should indicate which one has a longer duration
    
    :param mode: the mode of generating. Options: 'random', 'random_validate', 'test'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param prod_interval1, prod_interval2, dly_interval: used only for 'test' mode
    
    :return: Trial object
    '''
    dt = config['dt']
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # TRY DATASET FIRST - ADD THIS BLOCK:
        dataset_loader = get_dataset_loader('interval_comparison')
        if dataset_loader and dataset_loader.train_data is not None:
            params = dataset_loader.get_next_params(batch_size)
            prod_interval1 = (np.array([p['prod_interval1'] for p in params]) / dt).astype(int)
            prod_interval2 = (np.array([p['prod_interval2'] for p in params]) / dt).astype(int)
            dly_interval = (np.array([p['dly_interval'] for p in params]) / dt).astype(int)
            if dataset_loader.total_samples_used % 10000 == 0:  # Print every 10k samples
                print(f"Dataset intervals (trial {dataset_loader.total_samples_used}):")
                print(f"  Interval1: {params[0]['prod_interval1']:.1f}ms ({prod_interval1[0]} steps)")
                print(f"  Interval2: {params[0]['prod_interval2']:.1f}ms ({prod_interval2[0]} steps)")
                print(
                    f"  Sequence length will be: ~{(prod_interval1[0] + prod_interval2[0] + dly_interval[0] + 100)} steps")
        else:
            # FALLBACK TO ORIGINAL CODE:
            prod_interval1 = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
            prod_interval2 = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
            dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # TRY DATASET FIRST - ADD THIS BLOCK:
        dataset_loader = get_dataset_loader('interval_comparison')
        if dataset_loader and dataset_loader.train_data is not None:
            params = dataset_loader.get_next_params(batch_size)
            prod_interval1 = (np.array([p['prod_interval1'] for p in params]) / dt).astype(int)
            prod_interval2 = (np.array([p['prod_interval2'] for p in params]) / dt).astype(int)
            dly_interval = (np.array([p['dly_interval'] for p in params]) / dt).astype(int)
            if dataset_loader.total_samples_used % 10000 == 0:  # Print every 10k samples
                print(f"Dataset intervals (trial {dataset_loader.total_samples_used}):")
                print(f"  Interval1: {params[0]['prod_interval1']:.1f}ms ({prod_interval1[0]} steps)")
                print(f"  Interval2: {params[0]['prod_interval2']:.1f}ms ({prod_interval2[0]} steps)")
                print(
                    f"  Sequence length will be: ~{(prod_interval1[0] + prod_interval2[0] + dly_interval[0] + 100)} steps")
        else:
            # FALLBACK TO ORIGINAL CODE:
            prod_interval1 = (rng.uniform(400, 1400, batch_size) / dt).astype(int)
            prod_interval2 = (rng.uniform(400, 1400, batch_size) / dt).astype(int)
            dly_interval = (rng.uniform(600, 1600, batch_size) / dt).astype(int)

        #prod_interval1 = (rng.uniform(600, 1200, batch_size)/dt).astype(int)
        #prod_interval2 = (rng.uniform(600, 1200, batch_size)/dt).astype(int)

        #dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval1 = kwargs['prod_interval1']
        if not hasattr(prod_interval1, '__iter__'):
            prod_interval1 = np.array([prod_interval1] * batch_size)
        prod_interval1 = (prod_interval1 / dt).astype(int)

        prod_interval2 = kwargs['prod_interval2']
        if not hasattr(prod_interval2, '__iter__'):
            prod_interval2 = np.array([prod_interval2] * batch_size)
        prod_interval2 = (prod_interval2 / dt).astype(int)

        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(100, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus (duration = prod_interval1)
    stim1_off = stim1_on + prod_interval1
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus (duration = prod_interval2)
    stim2_off = stim2_on + prod_interval2
    # total time steps
    xtdim = stim2_off + response_duration

    # output: which interval was longer
    output_target1 = 1. * (prod_interval1 > prod_interval2)  # 1 if first interval longer
    output_target2 = 1. * (prod_interval1 <= prod_interval2) # 1 if second interval longer
    # Add this debug code after the target calculation lines:
    if batch_size <= 8:  # Only for small batches to avoid spam
        print(f"\nDEBUG TARGET GENERATION:")
        for i in range(min(3, batch_size)):
            print(f"  Trial {i}: I1={prod_interval1[i] * dt:.0f}ms, I2={prod_interval2[i] * dt:.0f}ms")
            print(f"    I1 > I2: {prod_interval1[i] > prod_interval2[i]}")
            print(f"    Target1: {output_target1[i]}, Target2: {output_target2[i]}")
            print(f"    Expected: {'[1,0]' if prod_interval1[i] > prod_interval2[i] else '[0,1]'}")

    trial = Trial(config, xtdim.max(), batch_size)
    # First stimulus on input channel 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    # Second stimulus on input channel 1
    trial.add('input', 1, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))

    # Output - which interval was longer
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)
    # Cost mask for both output channels
    trial.add('cost_mask', 0, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'delay': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, xtdim)}

    trial.prod_interval1 = prod_interval1
    trial.prod_interval2 = prod_interval2
    trial.dly_interval = dly_interval
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def interval_comparison(config, mode, **kwargs):
    '''Wrapper function for interval comparison task'''
    return _interval_comparison(config, mode, **kwargs)


def _time_bisection(config, mode, **kwargs):
    """Time Bisection Task - with corrected debugging"""
    dt = config.get('dt', 20)
    rng = config.get('rng', 100)
    response_duration = int(kwargs.get('response_duration', 300) / dt)

    short_standard = kwargs.get('short_standard', 1000)
    long_standard = kwargs.get('long_standard', 2000)
    std = kwargs.get('std', 40)
    batch_size = kwargs.get('batch_size', 8)

    # DEBUG: Print target generation info
    #print(f"\nDEBUG TIME_BISECTION TARGET GENERATION:")
    #print(f"  Mode: {mode}")
    #print(f"  Batch size: {batch_size}")
    #print(f"  Short standard: {short_standard}")
    #print(f"  Long standard: {long_standard}")

    # Get dataset loader (this was missing from the debug code)
    dataset_loader = get_dataset_loader('time_bisection')

    if mode == 'random':
        if dataset_loader and dataset_loader.train_data is not None:
            #print(f"  Using DATASET targets (correct_choice field)")
            params = dataset_loader.get_next_params(batch_size)
            prod_interval = np.array([p['duration'] for p in params])
            prod_interval = (prod_interval / dt).astype(int)

            # Use dataset targets directly
            correct_choices = np.array([p['correct_choice'] for p in params])
            output_target1 = (correct_choices == 0).astype(np.float32)
            output_target2 = (correct_choices == 1).astype(np.float32)

            # Print first few examples
            #for i, p in enumerate(params[:3]):
                #print(f"    Sample {i}: duration={p['duration']:.1f}ms, correct_choice={p['correct_choice']}")
        else:
            #print(f"  Using FALLBACK target generation (closest standard)")
            standards = np.random.choice([short_standard, long_standard], size=batch_size)
            prod_interval = np.random.normal(loc=standards, scale=std, size=batch_size)
            prod_interval = np.maximum(prod_interval, 0)

            # Use closest-standard logic
            short_distances = np.abs(prod_interval - short_standard)
            long_distances = np.abs(prod_interval - long_standard)
            output_target1 = (short_distances < long_distances).astype(np.float32)
            output_target2 = (short_distances >= long_distances).astype(np.float32)

            prod_interval = (prod_interval / dt).astype(int)

    elif mode == 'random_validate':
        if dataset_loader and dataset_loader.train_data is not None:
            #print(f"  Using DATASET targets (correct_choice field)")
            params = dataset_loader.get_next_params(batch_size)
            prod_interval = np.array([p['duration'] for p in params])
            prod_interval = (prod_interval / dt).astype(int)

            correct_choices = np.array([p['correct_choice'] for p in params])
            output_target1 = (correct_choices == 0).astype(np.float32)
            output_target2 = (correct_choices == 1).astype(np.float32)
        else:
            #print(f"  Using FALLBACK target generation (standards)")
            prod_interval = np.random.choice([short_standard, long_standard], size=batch_size)

            short_distances = np.abs(prod_interval - short_standard)
            long_distances = np.abs(prod_interval - long_standard)
            output_target1 = (short_distances < long_distances).astype(np.float32)
            output_target2 = (short_distances >= long_distances).astype(np.float32)

            prod_interval = (prod_interval / dt).astype(int)

    elif mode == 'test':
        print(f"  Using TEST mode target generation")
        batch_size = kwargs['batch_size']

        if 'duration' in kwargs:
            prod_interval = kwargs['duration']
            if not hasattr(prod_interval, '__iter__'):
                prod_interval = np.array([prod_interval] * batch_size)
        else:
            possible_intervals = np.arange(short_standard, long_standard + 1, 10)
            prod_interval = np.random.choice(possible_intervals, size=batch_size)

        # Generate correct targets for test data using closest standard logic
        short_distances = np.abs(prod_interval - short_standard)
        long_distances = np.abs(prod_interval - long_standard)
        output_target1 = (short_distances < long_distances).astype(np.float32)
        output_target2 = (short_distances >= long_distances).astype(np.float32)

        print(f"  Test duration: {prod_interval[0] if len(prod_interval) > 0 else 'none'}ms")
        print(
            f"  Test target: class {np.argmax([output_target1[0], output_target2[0]]) if len(output_target1) > 0 else 'none'}")

        prod_interval = (prod_interval / dt).astype(int)
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # Print final targets
    #print(f"  Final targets - Class 0 count: {output_target1.sum()}, Class 1 count: {output_target2.sum()}")

    # Validation
    target_sum = output_target1 + output_target2
    if not np.allclose(target_sum, 1.0):
        print(f"  ERROR: Target sums are wrong: {target_sum}")
        raise ValueError(f"Target generation failed! Target sums: {target_sum}")

    # Rest of the function continues unchanged...
    dly_interval = 300
    if not hasattr(dly_interval, '__iter__'):
        dly_interval = np.random.choice([dly_interval], size=batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    stim_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)
    stim_off = stim_on + prod_interval
    response_on = stim_off + dly_interval
    xtdim = (response_on + response_duration).astype(int)

    trial = Trial(config, xtdim.max(), batch_size)
    trial.comparison_intervals = prod_interval * dt

    trial.add('input', 0, ons=stim_on, offs=stim_off, strengths=trial.expand(1.))
    trial.add('input', 1, ons=response_on, offs=xtdim, strengths=trial.expand(1.))

    trial.add('out', 0, ons=response_on, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=xtdim, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))

    trial.epochs = {
        'fix': (None, stim_on),
        'stimulus': (stim_on, stim_off),
        'delay': (stim_off, stim_off + dly_interval),
        'go': (response_on, xtdim)
    }

    trial.short_standard = short_standard
    trial.long_standard = long_standard
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def time_bisection(config, mode, **kwargs):
    """Wrapper function for time bisection task"""
    return _time_bisection(config, mode, **kwargs)


def _gaussian_bisection(config, mode, **kwargs):
    """
    Time Bisection Task: The network is presented with a stimulus for a specific duration
    and must classify it as either "short" (closer to short_standard) or "long" (closer to long_standard).
    """

    # rng = config['rng']
    std = kwargs.get('std', 10)
    short_standard = kwargs.get('short_standard',
                                300)  # Should be named "Low/High standard" because we speak about gaussians and not times, but
    long_standard = kwargs.get('long_standard', 900)  # compatability is more important..

    batch_size = kwargs.get('batch_size', 8)  # Number of trials to generate in this batch
    mode = 'random'

    if mode == 'random':
        # here there should be generated a number from one of the gaussians
        standards = np.random.choice([short_standard, long_standard], size=batch_size)
        prod_interval = np.random.normal(loc=standards, scale=std, size=batch_size)
        # print(f'gaussian production interval:\n {prod_interval}')
    elif mode == 'random_validate':
        # Generate prod_interval1: one of [300, 900] ms for each batch element
        prod_interval = np.random.choice([short_standard, long_standard],
                                         size=batch_size)  # Creates an array sized batch_size that each entrance is either 300 or 900.

    elif mode == 'test':
        # Here we can conisder changing the std for the test. Anyhow, need to discuss.
        # Now we just generate uniformly distributed values between standard, which can be more related to real life task.
        possible_intervals = np.arange(short_standard, long_standard + 1, 10)
        prod_interval = np.random.choice(possible_intervals, size=batch_size)


    else:
        raise ValueError('Unknown mode: ' + str(mode))

        # Set trial to have 10 timesteps for each batch element
    xtdim = np.full(batch_size, 10, dtype=int)
    #     xtdim = np.ones(batch_size, dtype=int)
    trial = Trial(config, xtdim.max(), batch_size)  # Create trial object

    stim_on = np.zeros(batch_size, dtype=int)  # Starts at time 0 for all trials
    stim_off = np.full(batch_size, 10, dtype=int)  # Ends at time 10 for all trials
    # stim_off = np.ones(batch_size, dtype=int)
    trial.comparison_intervals = prod_interval
    # Input channel 0: stimulus with the scalar value from prod_interval for all 10 timesteps
    trial.add('input', 0, ons=stim_on, offs=stim_off, strengths=prod_interval)
    #    trial.add('input', 0, ons=stim_on, offs=stim_off, strengths=trial.expand(1.))

    # output: which interval was longer
    output_target1 = 1. * (prod_interval <= 600)  # SHORT INTERVAL
    output_target2 = 1 - output_target1  # LONG INTERVAL
    trial.add('out', 0, ons=stim_on, offs=stim_off, strengths=output_target1)
    trial.add('out', 1, ons=stim_on, offs=stim_off, strengths=output_target2)
    # print("stim_on",stim_on)
    # print("stim_off", stim_off)
    trial.add('cost_mask', 0, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))

    # Epochs are for visual inspection and logging/debugging
    trial.epochs = {
        'fix': (None),
        'stimulus': (stim_on, stim_off),
        'delay': (None),
        'go': (stim_on, stim_off)
    }

    trial.short_standard = short_standard
    trial.long_standard = long_standard
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def gaussian_bisection(config, mode, **kwargs):
    """Wrapper function for time bisection task"""
    return _gaussian_bisection(config, mode, **kwargs)


# Map task names to their corresponding functions
rule_mapping = {
    'interval_production': interval_production,
    'interval_comparison': interval_comparison,
    'time_bisection': time_bisection,
    'gaussian_bisection': gaussian_bisection
}



def generate_trials(rule, config, mode, noise_on=True, **kwargs):
    """Generate one batch of data for a specified task.
    
    Args:
        rule: str, the name of the task
        config: dictionary of hyperparameters
        mode: str, the mode of generating ('random', 'random_validate', 'test')
        noise_on: bool, whether input noise is given
        **kwargs: Additional task-specific parameters
        
    Returns:
        trial: Trial class instance containing input and target output patterns
    """
    #print(f"DEBUG: generate_trials called with rule={rule}")
    #print(f"DEBUG: rule_mapping contains: {list(rule_mapping.keys())}")
    #print(f"DEBUG: About to call rule_mapping[{rule}]")
    # NEW: Check what function is actually in rule_mapping
    func = rule_mapping[rule]
    #print(f"DEBUG: Function name: {func.__name__}")
    #print(f"DEBUG: Function module: {func.__module__}")
    #print(f"DEBUG: Function id: {id(func)}")
    if rule not in rule_mapping:
        raise ValueError(f"Unknown rule: {rule}. Available rules: {list(rule_mapping.keys())}")
    
    # Pass noise_on parameter to the task function
    kwargs['noise_on'] = noise_on
    
    # Generate the trial using the appropriate task function
    trial = rule_mapping[rule](config, mode, **kwargs)
    
    # Add noise to inputs if requested
    if noise_on:
        trial.add_x_noise()
    
    return trial


def get_default_input_output_sizes(rule_name):
    """Return the default input and output sizes for each task.
    
    Args:
        rule_name: str, name of the task
        
    Returns:
        tuple (n_input, n_output): Number of input and output channels needed
    """
    if rule_name == 'interval_production':
        return 2, 1  # 2 input channels, 1 output channel
    elif rule_name == 'interval_comparison':
        return 2, 2  # 2 input channels, 2 output channels
    elif rule_name == 'time_bisection':
        return 2, 2  # 2 input channel, 2 output channels
    elif rule_name == 'gaussian_bisection':
        return 1, 2
    else:
        raise ValueError(f"Unknown rule: {rule_name}") 