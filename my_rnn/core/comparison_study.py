"""
Amended Concurrent Model Training Comparison - 2M Trials
========================================================

Key features:
1. Larger intervals (1000-2000ms for heartbeat coverage)
2. 2 million trial training
3. Cyclical piezo analysis throughout slices
4. Eigenvalue analysis at end
5. Time delay implementation
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Patch task generation for larger intervals
import task

# Override task functions with larger intervals
def large_interval_production(config, mode, **kwargs):
    """Interval production with 1000-2000ms intervals"""
    dt = config['dt']
    rng = config.get('rng', np.random.RandomState())
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode in ['random', 'random_validate']:
        batch_size = kwargs['batch_size']
        # LARGER INTERVALS: 1000-2000ms instead of 400-1400ms
        prod_interval = (rng.uniform(1000, 2000, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(800, 1200, batch_size)/dt).astype(int)
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

    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    stim1_off = stim1_on + pulse_duration
    stim2_on = stim1_off + prod_interval
    stim2_off = stim2_on + pulse_duration
    control_on = stim2_off + dly_interval
    control_off = control_on + pulse_duration
    response_on = control_off + prod_interval
    response_off = response_on + response_duration
    xtdim = response_off

    trial = task.Trial(config, xtdim.max(), batch_size)
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    trial.add('out', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on), 'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on), 'stim2': (stim2_on, stim2_off),
                    'delay': (stim2_off, control_on), 'go_cue': (control_on, control_off),
                    'go': (control_off, response_on), 'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.dly_interval = dly_interval
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()
    return trial

def large_interval_comparison(config, mode, **kwargs):
    """Interval comparison with 1000-2000ms intervals"""
    dt = config['dt']
    rng = config['rng']
    response_duration = int(300/dt)

    if mode in ['random', 'random_validate']:
        batch_size = kwargs['batch_size']
        # LARGER INTERVALS: 1000-2000ms
        prod_interval1 = (rng.uniform(1000, 2000, batch_size)/dt).astype(int)
        prod_interval2 = (rng.uniform(1000, 2000, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(800, 1200, batch_size)/dt).astype(int)
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

    if kwargs['noise_on']:
        stim1_on = (rng.uniform(100, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    stim1_off = stim1_on + prod_interval1
    stim2_on = stim1_off + dly_interval
    stim2_off = stim2_on + prod_interval2
    xtdim = stim2_off + response_duration

    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    trial = task.Trial(config, xtdim.max(), batch_size)
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    trial.add('input', 1, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)
    trial.add('cost_mask', 0, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on), 'stim1': (stim1_on, stim1_off),
                    'delay': (stim1_off, stim2_on), 'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, xtdim)}

    trial.prod_interval1 = prod_interval1
    trial.prod_interval2 = prod_interval2
    trial.dly_interval = dly_interval
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()
    return trial

def large_time_bisection(config, mode, **kwargs):
    """Time bisection with 800ms/1600ms standards"""
    dt = config['dt']
    rng = config['rng']
    response_duration = int(kwargs.get('response_duration', 300)/dt)

    # LARGER STANDARDS: 800ms and 1600ms instead of 300ms and 900ms
    short_standard = kwargs.get('short_standard', 800)
    long_standard = kwargs.get('long_standard', 1600)
    batch_size = kwargs.get('batch_size', 512)

    if mode in ['random', 'random_validate']:
        prod_interval = np.random.choice([short_standard, long_standard], size=batch_size)
        prod_interval = (prod_interval / dt).astype(int)
    elif mode == 'test':
        if kwargs.get('test_type') == 'uniformly distributed':
            possible_intervals = np.arange(short_standard, long_standard + 1, 20)
            prod_interval = np.random.choice(possible_intervals, size=batch_size)
            prod_interval = (prod_interval / dt).astype(int)

    dly_interval = 0
    if not hasattr(dly_interval, '__iter__'):
        dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    stim_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)
    stim_off = stim_on + prod_interval
    response_on = stim_off + dly_interval
    xtdim = (response_on + response_duration).astype(int)

    trial = task.Trial(config, xtdim.max(), batch_size)
    trial.comparison_intervals = prod_interval * dt

    trial.add('input', 0, ons=stim_on, offs=stim_off, strengths=trial.expand(1.))
    trial.add('input', 1, ons=response_on, offs=xtdim, strengths=trial.expand(1.))

    midpoint = (short_standard + long_standard) / 2  # 1200ms
    output_target1 = 1. * (prod_interval * dt <= midpoint)
    output_target2 = 1. * (prod_interval * dt > midpoint)

    trial.add('out', 0, ons=response_on, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=xtdim, strengths=output_target2)
    trial.add('cost_mask', 0, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim_on, offs=xtdim, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim_on), 'stimulus': (stim_on, stim_off),
                    'delay': (stim_off, response_on), 'go': (response_on, xtdim)}

    trial.short_standard = short_standard
    trial.long_standard = long_standard
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()
    return trial

# Patch the task functions - focusing on interval_production and interval_comparison
# (time_bisection function included but not used by default)
task.interval_production = large_interval_production
task.interval_comparison = large_interval_comparison
task.time_bisection = large_time_bisection
task.rule_mapping = {
    'interval_production': large_interval_production,
    'interval_comparison': large_interval_comparison,
    'time_bisection': large_time_bisection
}

print("✅ Task intervals patched for larger heartbeat-covering intervals")
print("🎯 Focus: interval_production and interval_comparison tasks")

# Main imports
import train
import default
import network
import tools

def generate_cardiac_data(length):
    """Generate realistic cardiac pressure with clear cycles"""
    t = np.linspace(0, length / 60, length)
    cardiac_freq = 1.0  # 60 BPM

    # Clear cardiac pattern
    cardiac_cycle = np.sin(2 * np.pi * cardiac_freq * t) ** 3
    diastolic = 0.3 * np.sin(4 * np.pi * cardiac_freq * t)

    # R-wave spikes
    r_interval = int(60 / cardiac_freq)
    r_positions = np.arange(0, length, r_interval)
    r_waves = np.zeros(length)
    for pos in r_positions:
        if pos < length:
            spike_width = 3
            start = max(0, pos - spike_width)
            end = min(length, pos + spike_width + 1)
            r_waves[start:end] += np.exp(-((np.arange(start, end) - pos) ** 2) / (spike_width / 2))

    pressure = cardiac_cycle + diastolic + 0.5 * r_waves
    pressure += 0.05 * np.random.randn(len(t))
    pressure = pressure - pressure.min() + 0.2
    pressure = pressure * 1.5 + 0.5

    return pressure, r_positions

def create_hb_sequence(pressure, r_peaks):
    """Create heartbeat sequence"""
    ecg_col = np.zeros_like(pressure)
    rpeak_col = np.zeros_like(pressure)

    for pos in r_peaks:
        if pos < len(rpeak_col):
            rpeak_col[pos] = 1

    return np.stack([ecg_col, pressure, rpeak_col], axis=1)

def compute_eigenvalues(weight_matrix):
    """Compute eigenvalues of weight matrix"""
    if isinstance(weight_matrix, torch.Tensor):
        weight_matrix = weight_matrix.detach().cpu().numpy()
    return np.linalg.eigvals(weight_matrix)

def plot_eigenvalues(standard_eigs, piezo_eigs, save_path):
    """Plot eigenvalues in complex plane"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Standard model
    ax1.scatter(standard_eigs.real, standard_eigs.imag, alpha=0.6, s=20, color='blue')
    ax1.set_xlabel('Real Component')
    ax1.set_ylabel('Imaginary Component')
    ax1.set_title('Standard Model Eigenvalues')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
    ax1.legend()

    # Piezo model
    ax2.scatter(piezo_eigs.real, piezo_eigs.imag, alpha=0.6, s=20, color='red')
    ax2.set_xlabel('Real Component')
    ax2.set_ylabel('Imaginary Component')
    ax2.set_title('Piezo Model Eigenvalues')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
    ax2.legend()

    # Overlay
    ax3.scatter(standard_eigs.real, standard_eigs.imag, alpha=0.6, s=20,
                color='blue', label='Standard')
    ax3.scatter(piezo_eigs.real, piezo_eigs.imag, alpha=0.6, s=20,
                color='red', label='Piezo')
    ax3.set_xlabel('Real Component')
    ax3.set_ylabel('Imaginary Component')
    ax3.set_title('Eigenvalue Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Stability boundary')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_piezo_cycles(piezo_trainer, timesteps=1000):
    """Analyze how piezo changes throughout slices"""
    if not hasattr(piezo_trainer.model, 'piezo') or piezo_trainer.model.piezo is None:
        return None

    # Generate cardiac data
    pressure, r_peaks = generate_cardiac_data(timesteps)
    hb_sequence, _ = create_hb_sequence(pressure, r_peaks)
    hb_tensor = torch.tensor(hb_sequence, dtype=torch.float32)

    if piezo_trainer.is_cuda:
        hb_tensor = hb_tensor.cuda()

    # Extract responses AND track pressure values used by piezo
    slice_size = piezo_trainer.model.piezo.slice_size
    T = timesteps // slice_size

    pressure_slices, min_pressure = piezo_trainer.model.extract_pressure_slices(hb_tensor, T)
    piezo_responses = []
    slice_mean_pressures = []  # Track mean pressure per slice (what piezo actually uses)

    for slice_data in pressure_slices:
        # Calculate the mean pressure that piezo uses for this slice
        mean_pressure = torch.mean(slice_data).item()
        slice_mean_pressures.append(mean_pressure)

        # Get piezo response to this slice
        response = piezo_trainer.model.piezo(slice_data, min_pressure)
        piezo_responses.append(response.cpu().numpy())

    piezo_responses = np.array(piezo_responses)
    slice_mean_pressures = np.array(slice_mean_pressures)
    mean_response = np.mean(piezo_responses, axis=1)

    # Get only the connected piezo neurons for cleaner visualization
    connected_indices = piezo_trainer.model.piezo.connected_indices.cpu().numpy()
    connected_responses = piezo_responses[:, connected_indices]

    return {
        'pressure': pressure,
        'r_peaks': r_peaks,
        'piezo_responses': piezo_responses,
        'connected_responses': connected_responses,
        'mean_response': mean_response,
        'slice_mean_pressures': slice_mean_pressures,  # NEW: Mean pressure per slice
        'min_pressure': min_pressure.item(),           # NEW: Sequence minimum
        'slice_size': slice_size,
        'connected_indices': connected_indices,
        'timesteps': timesteps,
        'T': T
    }


def plot_piezo_analysis_interactive(analysis, trial_num=None):
    """Create interactive visualization that shows immediately"""
    if analysis is None:
        print("No analysis data to plot")
        return

    try:
        # Use interactive backend
        import matplotlib.pyplot as plt

        # Create figure with 4 subplots (2x2 grid for simplicity)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Time axes
        pressure_time = np.arange(len(analysis['pressure'])) / 60  # Convert to seconds
        slice_time = np.arange(analysis['T']) * analysis['slice_size'] / 60  # Slice times in seconds

        # 1. Original cardiac pressure signal
        ax1.plot(pressure_time, analysis['pressure'], 'b-', linewidth=1.5, label='Cardiac Pressure')
        # Mark R-peaks
        for r_peak in analysis['r_peaks']:
            if r_peak < len(pressure_time):
                ax1.axvline(x=pressure_time[r_peak], color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Pressure')
        ax1.set_title('A) Original Cardiac Pressure Signal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Heatmap of connected piezo neurons
        im = ax2.imshow(analysis['connected_responses'].T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Time Slices')
        ax2.set_ylabel('Connected Piezo Neurons')
        ax2.set_title('B) Piezo Neuron Activation Heatmap')
        plt.colorbar(im, ax=ax2, label='Activation Level')

        # 3. Mean piezo response over time
        ax3.plot(slice_time, analysis['mean_response'], 'g-', linewidth=2, label='Mean Piezo Response')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Mean Activation')
        ax3.set_title('C) Mean Piezo Response Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Pressure-to-Gain Analysis
        ax4.plot(slice_time, analysis['slice_mean_pressures'], 'b-', linewidth=2, label='Slice Mean Pressure',
                 alpha=0.8)
        ax4.plot(slice_time, analysis['mean_response'], 'r-', linewidth=2, label='Piezo Gain Response', alpha=0.8)
        correlation = np.corrcoef(analysis['slice_mean_pressures'], analysis['mean_response'])[0, 1]
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title(f'D) Pressure → Gain (r={correlation:.3f})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Overall title
        title_text = 'Piezo Response Analysis (Interactive)'
        if trial_num is not None:
            title_text += f' - Trial {trial_num:,}'
        plt.suptitle(title_text, fontsize=14)

        plt.tight_layout()

        # Show the plot immediately
        plt.show()

        # Print numerical summary
        print(f"\n🔬 Piezo Response Summary:")
        print(f"   Connected neurons: {len(analysis['connected_indices'])}")
        print(f"   Mean response: {analysis['mean_response'].mean():.4f}")
        print(f"   Response variability: {analysis['mean_response'].std():.4f}")
        print(f"   🎯 PRESSURE-GAIN CORRELATION: {correlation:.4f}")

        return fig

    except Exception as e:
        print(f"❌ Interactive plotting failed: {e}")
        return None

class AmendedConcurrentTrainer:
    """2 million trial concurrent trainer with all amendments"""

    def __init__(self, config):
        self.config = config
        self.rule_name = config['rule_name']
        self.w2 = config['w2_reg']
        self.r2 = config['r2_reg']
        self.device = config['device']
        self.max_trials = config['max_trials']  # 2 million
        self.display_step = config['display_step']
        self.base_dir = config['base_dir']
        self.is_cuda = (self.device == 'cuda')

        tools.mkdir_p(self.base_dir)

        self.progress = {
            'standard': {'trials': [], 'costs': [], 'success_prob': [], 'choice_error': []},
            'piezo': {'trials': [], 'costs': [], 'success_prob': [], 'choice_error': []}
        }

        print(f"🔬 Amended Concurrent Training: {self.max_trials:,} trials")

    def initialize_trainer(self, use_piezo):
        """Initialize trainer with amended settings"""
        model_type = "PIEZO" if use_piezo else "STANDARD"

        for attempt in range(10):
            seed = np.random.randint(0, 1000000)
            np.random.seed(seed)
            torch.manual_seed(seed)

            hp = default.get_default_hp(self.rule_name, use_piezo=use_piezo, random_seed=seed)
            # STABILIZATION FOR LARGE INTERVALS
            hp['initial_std'] = 0.05  # Much smaller!
            hp['learning_rate'] = 0.0001  # Smaller learning rate
            hp['sigma_rec'] = 0.01  # Less recurrent noise
            hp['l2_weight'] = self.w2
            hp['l2_firing_rate'] = self.r2

            if use_piezo:
                hp['use_temporal_delay'] = True  # Enable time delay
                hp['temporal_delay_steps'] = 3
                hp['piezo_connection_fraction'] = 0.15

            model_suffix = "piezo" if use_piezo else "standard"
            model_dir = os.path.join(self.base_dir, f"{self.rule_name}_{model_suffix}")
            tools.mkdir_p(model_dir)

            trainer = train.Trainer(rule_name=self.rule_name, hp=hp, model_dir=model_dir, is_cuda=self.is_cuda)

            try:
                sample_batched = next(iter(trainer.dataloader_train))
                if self.is_cuda:
                    sample_batched['inputs'] = sample_batched['inputs'].cuda()
                    sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                    sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                    sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                    sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

                sample_batched['rule_name'] = self.rule_name
                trainer.train_stepper.stepper(**sample_batched)
                cost = trainer.train_stepper.cost.item()

                if torch.isfinite(torch.tensor(cost)):
                    print(f"✅ {model_type} initialized: cost {cost:.6f}")
                    return trainer
                else:
                    print(f"💥 Non-finite cost, retrying...")
                    continue
            except Exception as e:
                print(f"💥 Exception: {e}, retrying...")
                continue

        raise RuntimeError(f"Failed to initialize {model_type}")

    def train_concurrent(self):
        """Main 2M trial concurrent training"""
        print("🔧 Initializing models...")
        standard_trainer = self.initialize_trainer(use_piezo=False)
        piezo_trainer = self.initialize_trainer(use_piezo=True)

        print(f"⚡ Starting 2M trial training...")

        standard_dataloader = iter(standard_trainer.dataloader_train)
        piezo_dataloader = iter(piezo_trainer.dataloader_train)

        start_time = time.time()
        trial = 0

        while trial < self.max_trials:
            # Alternate training steps
            for trainer_name, trainer, dataloader_iter in [
                ('standard', standard_trainer, standard_dataloader),
                ('piezo', piezo_trainer, piezo_dataloader)
            ]:
                try:
                    sample_batched = next(dataloader_iter)
                except StopIteration:
                    if trainer_name == 'standard':
                        standard_dataloader = iter(standard_trainer.dataloader_train)
                        sample_batched = next(standard_dataloader)
                    else:
                        piezo_dataloader = iter(piezo_trainer.dataloader_train)
                        sample_batched = next(piezo_dataloader)

                if self.is_cuda:
                    sample_batched['inputs'] = sample_batched['inputs'].cuda()
                    sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                    sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                    sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                    sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

                sample_batched['rule_name'] = self.rule_name
                trainer.train_stepper.stepper(**sample_batched)
                trial += 1

            # Periodic evaluation
            if trial % self.display_step == 0:
                current_trials = trial * standard_trainer.hp['batch_size_train']
                print(f"\nTrial {trial} ({current_trials:,} samples):")

                for trainer_name, trainer in [('standard', standard_trainer), ('piezo', piezo_trainer)]:
                    trainer.log['trials'].append(current_trials)
                    trainer.log['times'].append(time.time() - start_time)

                    cost, success_prob, choice_error = trainer.do_eval()

                    self.progress[trainer_name]['trials'].append(current_trials)
                    self.progress[trainer_name]['costs'].append(cost)
                    self.progress[trainer_name]['success_prob'].append(success_prob)
                    self.progress[trainer_name]['choice_error'].append(choice_error)

                    print(f"  {trainer_name.upper()}: cost {cost:.6f}, success {success_prob:.3f}, error {choice_error:.3f}")

                # Cyclical analysis every 100k trials with visualization
                if trial % 100000 == 0:
                    try:
                        print(f"  🔬 Running cyclical analysis with visualization...")
                        analysis = analyze_piezo_cycles(piezo_trainer, timesteps=2000)
                        if analysis is not None:
                            # Create plot save path
                            plot_path = os.path.join(self.base_dir, f'piezo_analysis_trial_{trial:07d}.png')

                            # Generate comprehensive visualization
                            plot_piezo_analysis(analysis, save_path=plot_path, trial_num=trial)

                            print(f"  📊 Mean response: {np.mean(analysis['mean_response']):.4f}")
                            print(f"  📈 Active neurons: {np.sum(np.max(analysis['connected_responses'], axis=0) > 0.01)}/{analysis['connected_responses'].shape[1]}")
                    except Exception as e:
                        print(f"  ⚠️ Cyclical analysis failed: {e}")

        # FINAL ANALYSIS
        print(f"\n📊 FINAL ANALYSIS (2M trials complete)")

        # Eigenvalue analysis
        try:
            standard_eigs = compute_eigenvalues(standard_trainer.model.weight_hh)
            piezo_eigs = compute_eigenvalues(piezo_trainer.model.weight_hh)

            eigenvalue_path = os.path.join(self.base_dir, 'eigenvalue_analysis.png')
            plot_eigenvalues(standard_eigs, piezo_eigs, eigenvalue_path)

            print(f"✅ Eigenvalue analysis saved: {eigenvalue_path}")
            print(f"   Standard max real: {np.max(standard_eigs.real):.4f}")
            print(f"   Piezo max real: {np.max(piezo_eigs.real):.4f}")
        except Exception as e:
            print(f"⚠️ Eigenvalue analysis failed: {e}")

        # Final cyclical analysis with comprehensive visualization
        try:
            print("🔬 Generating final cyclical analysis with full visualization...")
            final_analysis = analyze_piezo_cycles(piezo_trainer, timesteps=3000)
            if final_analysis is not None:
                # Save final comprehensive plot
                final_plot_path = os.path.join(self.base_dir, 'final_piezo_analysis.png')
                plot_piezo_analysis(final_analysis, save_path=final_plot_path, trial_num=trial)

                print(f"✅ Final cyclical analysis: {len(final_analysis['mean_response'])} slices analyzed")
                print(f"📊 Final plot saved: {final_plot_path}")
        except Exception as e:
            print(f"⚠️ Final cyclical analysis failed: {e}")

        # Save results
        results = {'config': self.config, 'progress': self.progress}
        results_path = os.path.join(self.base_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n🎉 2M TRIAL TRAINING COMPLETE!")
        print(f"   Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")


def test_piezo_visualization():
    """Test the piezo visualization with a quick setup"""
    print("🧪 Testing piezo visualization...")

    try:
        # Quick test setup using interval_comparison
        import default
        hp = default.get_default_hp('interval_comparison', use_piezo=True)
        hp['use_temporal_delay'] = True
        hp['temporal_delay_steps'] = 3

        # Create a piezo-enabled trainer (without full training)
        trainer = train.Trainer(
            rule_name='interval_comparison',
            hp=hp,
            model_dir='test_piezo_viz',
            is_cuda=False
        )

        # Run analysis and visualization
        analysis = analyze_piezo_cycles(trainer, timesteps=1200)
        if analysis is not None:
            # Show the plot instead of just saving it
            plot_piezo_analysis_interactive(analysis, trial_num=0)
            plot_piezo_analysis(analysis, save_path='test_piezo_analysis.png', trial_num=0)
            print("✅ Test visualization complete! Check 'test_piezo_analysis.png'")
        else:
            print("❌ Test failed - no piezo interface found")

    except Exception as e:
        print(f"❌ Test visualization failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Let user choose between the two main timing tasks
    print("🚀 Amended 2M Trial Comparison Setup")
    print("Choose timing task:")
    print("1) interval_production - Learn to reproduce time intervals")
    print("   Network sees two pulses → learns to output after same interval")
    print("2) interval_comparison - Compare durations of two stimuli")
    print("   Network sees two intervals → decides which was longer")

    try:
        task_choice = input("\nEnter 1 or 2 (default: 2): ").strip()
        if task_choice == '1':
            rule_name = 'interval_production'
            print("📏 Selected: Interval Production (temporal reproduction)")
        else:
            rule_name = 'interval_comparison'  # Default
            print("⚖️ Selected: Interval Comparison (temporal discrimination)")
    except:
        rule_name = 'interval_comparison'  # Default fallback
        print("⚖️ Selected: Interval Comparison (default)")

    config = {
        'rule_name': rule_name,
        'w2_reg': 0.0001,
        'r2_reg': 0.0,
        'device': 'cpu',
        'max_trials': 2000000,  # 2 MILLION TRIALS
        'display_step': 1000,
        'base_dir': f'amended_2m_comparison_{rule_name}'
    }

    print(f"\n✅ Configuration:")
    print(f"   Task: {rule_name}")
    print(f"   Intervals: 1000-2000ms (covers full heartbeat cycles)")
    print(f"   Training: 2M trials with time delay and cyclical analysis")
    print("📊 Enhanced with comprehensive piezo visualizations")

    # Ask user if they want to run full training or just test visualization
    try:
        choice = input("\nChoose:\n1) Run full 2M trial training\n2) Test piezo visualization only\nEnter 1 or 2: ")

        if choice == '2':
            test_piezo_visualization()
            return
        elif choice != '1':
            print("Invalid choice, running full training...")
    except:
        print("Running full training...")

    trainer = AmendedConcurrentTrainer(config)
    trainer.train_concurrent()

if __name__ == '__main__':
    main()