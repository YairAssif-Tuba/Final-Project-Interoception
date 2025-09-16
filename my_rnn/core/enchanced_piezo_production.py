"""
Enhanced Piezo vs Non-Piezo Network Comparison Script - INTERVAL PRODUCTION VERSION
===============================================================================

This script is based on enhanced_piezo_comparison.py but modified for the interval production task.
All original functionality is preserved, but the task is changed from interval_comparison to interval_production.

CHANGES FROM ENHANCED_PIEZO_COMPARISON.PY:
+ Task changed to interval_production
+ Modified task generation for production intervals
+ Updated performance metrics for production task
+ All connectivity tracking and convergence analysis preserved

Usage:
    python enhanced_piezo_production.py [--full-training] [--time-delay] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict
import pickle
from scipy import stats
# Add project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import default
import train
import network
import tools
from cardiac_data import generate_simple_cardiac_pressure, create_sparse_hb_sequence
from simple_piezo import SimplePiezoInterface
import json


class DatasetLoader:
    def __init__(self, dataset_path):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.train_data = self.dataset['train']
        self.index = 0
        self.total_samples_used = 0
        print(f"Loaded dataset with {len(self.train_data)} training samples")

    def get_next_params(self, batch_size):
        params = []
        for _ in range(batch_size):
            params.append(self.train_data[self.index % len(self.train_data)])
            self.index += 1
            self.total_samples_used += 1

        if self.total_samples_used % 1000 == 0:
            print(f"DATASET VERIFICATION: Used {self.total_samples_used} samples")
        return params
class EnhancedPiezoProductionAnalyzer:
    """Enhanced analyzer - IDENTICAL to original with connectivity & convergence tracking added."""

    def __init__(self, output_dir="enhanced_piezo_production_results", use_time_delay=False, num_runs=10):
        self.output_dir = output_dir
        self.use_time_delay = use_time_delay
        self.num_runs = num_runs
        tools.mkdir_p(self.output_dir)

        # IDENTICAL enhanced intervals from original (adapted for production task)
        self.enhanced_intervals = {
            'min_interval': 1200,  # 1.2 seconds minimum
            'max_interval': 2400,  # 2.4 seconds maximum
            'delay_interval': 1000  # 1 second delay
        }

        # NEW: Performance thresholds for convergence analysis
        self.convergence_thresholds = {
            'accuracy_90': 0.9,
            'accuracy_95': 0.95,
            'valid_response_90': 0.9,
            'valid_response_95': 0.95
        }

        print(f"Enhanced Piezo Production Analyzer")
        print(f"Output directory: {self.output_dir}")
        print(f"Training runs per network type: {self.num_runs}")
        print(f"Enhanced intervals: {self.enhanced_intervals['min_interval']}-{self.enhanced_intervals['max_interval']}ms")
        print(f"Time delay: {'ENABLED' if use_time_delay else 'DISABLED'}")
        #dataset_path = "enhanced_interval_datasets/interval_production_dataset.json"
        #if os.path.exists(dataset_path):
            #self.dataset_loader = DatasetLoader(dataset_path)
            #self.using_dataset = True
        #else:
            #self.dataset_loader = None
            #self.using_dataset = False

    def create_enhanced_hp(self, use_piezo=False):
        """MODIFIED: Use interval_production instead of interval_comparison"""
        hp = default.get_default_hp('interval_production', use_piezo=use_piezo)

        # Enhanced intervals to contain full heartbeat cycles
        hp['enhanced_intervals'] = self.enhanced_intervals

        # Piezo-specific enhancements
        if use_piezo:
            hp['heartbeat_slice_size'] = 20
            hp['piezo_connection_fraction'] = 0.15
            hp['use_temporal_delay'] = self.use_time_delay
            hp['temporal_delay_steps'] = 3

        return hp

    def modify_task_generation(self):
        """MODIFIED: Use interval_production task generation"""
        import task

        # Store original function
        original_interval_production = task._interval_production

        def enhanced_interval_production(config, mode, **kwargs):
            """Enhanced interval production with larger intervals."""
            # Call original function first
            trial = original_interval_production(config, mode, **kwargs)
            dt = config['dt']
            batch_size = trial.batch_size
            if self.dataset_loader is not None:
                params = self.dataset_loader.get_next_params(batch_size)

                trial.prod_interval = np.array([p['prod_interval'] / dt for p in params]).astype(int)
                trial.dly_interval = np.array([p['dly_interval'] / dt for p in params]).astype(int)

            # If we have enhanced intervals config, override the intervals
            elif 'enhanced_intervals' in config:
                dt = config['dt']
                batch_size = trial.batch_size
                rng = config['rng']

                # Generate enhanced intervals
                if mode in ['random', 'random_validate']:
                    min_int = config['enhanced_intervals']['min_interval']
                    max_int = config['enhanced_intervals']['max_interval']
                    delay_int = config['enhanced_intervals']['delay_interval']

                    # Override with enhanced intervals for production task
                    trial.prod_interval = (rng.uniform(min_int, max_int, batch_size) / dt).astype(int)
                    trial.dly_interval = (rng.uniform(delay_int, delay_int + 400, batch_size) / dt).astype(int)

                    # Regenerate the trial with new intervals
                    trial = self._regenerate_trial_with_intervals(trial, config)

            return trial

        # Patch the function
        task._interval_production = enhanced_interval_production
        task.rule_mapping['interval_production'] = enhanced_interval_production

        print("Task generation modified for enhanced production intervals")

    def _regenerate_trial_with_intervals(self, trial, config):
        """MODIFIED: Regenerate trial for interval production task"""
        dt = config['dt']
        batch_size = trial.batch_size
        rng = config['rng']

        # Get the new intervals
        prod_interval = trial.prod_interval
        dly_interval = trial.dly_interval

        pulse_duration = int(60 / dt)
        response_duration = int(300 / dt)

        # Recalculate timings for production task
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)
        stim1_off = stim1_on + pulse_duration
        stim2_on = stim1_off + prod_interval
        stim2_off = stim2_on + pulse_duration
        control_on = stim2_off + dly_interval
        control_off = control_on + pulse_duration
        response_on = control_off + prod_interval
        response_off = response_on + response_duration
        xtdim = response_off

        # Create new trial with correct dimensions
        from task import Trial
        new_trial = Trial(config, xtdim.max(), batch_size)

        # Add inputs and outputs for production task
        new_trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=new_trial.expand(1.))
        new_trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=new_trial.expand(1.))
        new_trial.add('input', 1, ons=control_on, offs=control_off, strengths=new_trial.expand(1.))

        # Output target for production task
        new_trial.add('out', 0, ons=response_on, offs=response_off, strengths=new_trial.expand(1.))
        new_trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=new_trial.expand(1.))

        # Set epochs for production task
        new_trial.epochs = {
            'fix': (None, stim1_on),
            'stim1': (stim1_on, stim1_off),
            'interval': (stim1_off, stim2_on),
            'stim2': (stim2_on, stim2_off),
            'delay': (stim2_off, control_on),
            'go_cue': (control_on, control_off),
            'go': (control_off, response_on),
            'response': (response_on, response_off)
        }

        # Copy over the intervals
        new_trial.prod_interval = prod_interval
        new_trial.dly_interval = dly_interval
        new_trial.seq_len = xtdim
        new_trial.max_seq_len = xtdim.max()

        return new_trial

    # NEW ENHANCEMENT 1: Connectivity analysis methods (IDENTICAL to original)
    def extract_connectivity_statistics(self, model, phase='initial'):
        """Extract detailed piezo connectivity statistics."""
        if not hasattr(model, 'piezo_connectivity') or model.piezo_connectivity is None:
            return None

        connectivity_raw = model.piezo_connectivity.detach().cpu().numpy().copy()

        # Apply same transformation as in network.py (tanh + relu)
        connectivity_activated = np.maximum(0, np.tanh(connectivity_raw))

        stats_dict = {
            'phase': phase,
            'raw_weights': connectivity_raw,
            'activated_weights': connectivity_activated,
            'raw_mean': np.mean(connectivity_raw),
            'raw_std': np.std(connectivity_raw),
            'raw_min': np.min(connectivity_raw),
            'raw_max': np.max(connectivity_raw),
            'activated_mean': np.mean(connectivity_activated),
            'activated_std': np.std(connectivity_activated),
            'activated_min': np.min(connectivity_activated),
            'activated_max': np.max(connectivity_activated),
            'num_active': np.sum(connectivity_activated > 0.01),
            'num_total': len(connectivity_activated),
            'active_fraction': np.sum(connectivity_activated > 0.01) / len(connectivity_activated)
        }

        return stats_dict

    def calculate_connectivity_correlation(self, initial_stats, final_stats):
        """Calculate correlation between initial and final connectivity weights."""
        if initial_stats is None or final_stats is None:
            return None

        # Correlations for both raw and activated weights
        raw_corr = np.corrcoef(initial_stats['raw_weights'], final_stats['raw_weights'])[0, 1]
        activated_corr = np.corrcoef(initial_stats['activated_weights'], final_stats['activated_weights'])[0, 1]

        # Weight change statistics
        raw_change = final_stats['raw_weights'] - initial_stats['raw_weights']
        activated_change = final_stats['activated_weights'] - initial_stats['activated_weights']

        correlation_analysis = {
            'raw_correlation': raw_corr,
            'activated_correlation': activated_corr,
            'raw_weight_change_mean': np.mean(raw_change),
            'raw_weight_change_std': np.std(raw_change),
            'activated_weight_change_mean': np.mean(activated_change),
            'activated_weight_change_std': np.std(activated_change),
            'max_raw_change': np.max(np.abs(raw_change)),
            'max_activated_change': np.max(np.abs(activated_change))
        }

        return correlation_analysis

    def train_networks(self, max_samples=5e5):
        """IDENTICAL to original - train both piezo and non-piezo networks with enhanced tracking."""
        print(f"\nEnhanced Network Training (Interval Production)")
        print(f"   Max samples per run: {max_samples:,.0f}")
        print(f"   Runs per network type: {self.num_runs}")
        print(f"   Total training runs: {self.num_runs * 2}")
        print("=" * 60)

        # Modify task generation for enhanced intervals
        print("DEBUG: Using built-in task dataset loading (no patching required)")

        #self.modify_task_generation()

        results = {
            'no_piezo': {
                'runs': [],
                'successful_runs': 0,
                'failed_runs': 0,
                'total_training_time': 0
            },
            'piezo': {
                'runs': [],
                'successful_runs': 0,
                'failed_runs': 0,
                'total_training_time': 0
            }
        }

        # Train non-piezo networks
        print(f"\nTraining {self.num_runs} Non-Piezo Networks")
        print("-" * 40)

        for run_idx in range(self.num_runs):
            print(f"\nNon-Piezo Run {run_idx + 1}/{self.num_runs}")

            model_dir = os.path.join(self.output_dir, f"no_piezo_run_{run_idx + 1}")
            start_time = time.time()

            stat, trainer = self._train_with_enhanced_tracking(
                model_dir=model_dir,
                use_piezo=False,
                max_samples=max_samples,
                display_step=1000,
                run_idx=run_idx + 1
            )

            training_time = time.time() - start_time
            results['no_piezo']['total_training_time'] += training_time

            run_result = {
                'run_idx': run_idx + 1,
                'model_dir': model_dir,
                'trainer': trainer,
                'status': stat,
                'training_time': training_time,
                'hp': trainer.hp if trainer else None
            }

            results['no_piezo']['runs'].append(run_result)

            if stat == 'OK':
                results['no_piezo']['successful_runs'] += 1
                print(f"   Run {run_idx + 1} successful ({training_time:.1f}s)")
            else:
                results['no_piezo']['failed_runs'] += 1
                print(f"   Run {run_idx + 1} failed ({training_time:.1f}s)")

        # Train piezo networks
        print(f"\nTraining {self.num_runs} Piezo Networks (Time Delay: {self.use_time_delay})")
        print("-" * 40)

        for run_idx in range(self.num_runs):
            print(f"\nPiezo Run {run_idx + 1}/{self.num_runs}")

            model_dir = os.path.join(self.output_dir, f"piezo_run_{run_idx + 1}")
            start_time = time.time()

            stat, trainer = self._train_with_enhanced_tracking(
                model_dir=model_dir,
                use_piezo=True,
                max_samples=max_samples,
                display_step=1000,
                run_idx=run_idx + 1
            )

            training_time = time.time() - start_time
            results['piezo']['total_training_time'] += training_time

            run_result = {
                'run_idx': run_idx + 1,
                'model_dir': model_dir,
                'trainer': trainer,
                'status': stat,
                'training_time': training_time,
                'hp': trainer.hp if trainer else None
            }

            results['piezo']['runs'].append(run_result)

            if stat == 'OK':
                results['piezo']['successful_runs'] += 1
                print(f"   Run {run_idx + 1} successful ({training_time:.1f}s)")
            else:
                results['piezo']['failed_runs'] += 1
                print(f"   Run {run_idx + 1} failed ({training_time:.1f}s)")

        # Save training summary
        self._save_multiple_runs_summary(results)

        # Print final summary
        print(f"\nMultiple Training Runs Complete!")
        print(f"   Non-Piezo: {results['no_piezo']['successful_runs']}/{self.num_runs} successful")
        print(f"   Piezo: {results['piezo']['successful_runs']}/{self.num_runs} successful")
        print(f"   Total time: {(results['no_piezo']['total_training_time'] + results['piezo']['total_training_time']):.1f}s")

        return results

    def _train_with_enhanced_tracking(self, model_dir, use_piezo, max_samples, display_step, run_idx):
        """MODIFIED: Use interval_production task instead of interval_comparison."""
        import shutil

        attempt = 0

        # Keep attempting to train until successful (original behavior)
        while True:
            attempt += 1

            try:
                print(f"      Attempt {attempt}...")

                # Create fresh hyperparameters for each attempt (IDENTICAL to original)
                hp = self.create_enhanced_hp(use_piezo=use_piezo)

                # Create trainer (MODIFIED: Use interval_production)
                trainer = train.Trainer(
                    model_dir=model_dir,
                    rule_name='interval_production',  # CHANGED from interval_comparison
                    hp=hp,
                    is_cuda=False
                )

                # NEW ENHANCEMENT 1: Record initial connectivity statistics (piezo only)
                initial_connectivity = None
                if use_piezo:
                    if hasattr(trainer.model, 'piezo_connectivity') and trainer.model.piezo_connectivity is not None:
                        initial_connectivity = self.extract_connectivity_statistics(trainer.model, 'initial')
                        if initial_connectivity:
                            print(f"      Initial connectivity: {initial_connectivity['num_active']}/{initial_connectivity['num_total']} active, "
                                  f"mean={initial_connectivity['activated_mean']:.3f}")
                    else:
                        print(f"      No piezo_connectivity found in model")

                # Attempt training (IDENTICAL to original)
                stat = trainer.train(max_samples=max_samples, display_step=display_step)

                # Check if training was successful
                if stat == 'OK':
                    # NEW ENHANCEMENT 1: Record final connectivity statistics and correlation (piezo only)
                    final_connectivity = None
                    connectivity_correlation = None
                    if use_piezo and hasattr(trainer.model, 'piezo_connectivity') and trainer.model.piezo_connectivity is not None:
                        final_connectivity = self.extract_connectivity_statistics(trainer.model, 'final')
                        if final_connectivity and initial_connectivity:
                            connectivity_correlation = self.calculate_connectivity_correlation(
                                initial_connectivity, final_connectivity)

                            print(f"      Final connectivity: {final_connectivity['num_active']}/{final_connectivity['num_total']} active, "
                                  f"mean={final_connectivity['activated_mean']:.3f}")
                            print(f"      Correlations: raw={connectivity_correlation['raw_correlation']:.3f}, "
                                  f"activated={connectivity_correlation['activated_correlation']:.3f}")

                            # Save connectivity analysis
                            connectivity_data = {
                                'initial': initial_connectivity,
                                'final': final_connectivity,
                                'correlation': connectivity_correlation,
                                'run_idx': run_idx,
                                'use_time_delay': self.use_time_delay
                            }

                            connectivity_file = os.path.join(model_dir, 'connectivity_analysis.pkl')
                            with open(connectivity_file, 'wb') as f:
                                pickle.dump(connectivity_data, f)
                            print(f"      Connectivity analysis saved")

                    print(f"      Training successful on attempt {attempt}")
                    return stat, trainer
                else:
                    print(f"      Training failed on attempt {attempt}: {stat}")

                    # CLEANUP LOGIC (IDENTICAL to original cluster_training_unit)
                    if use_piezo:
                        # For piezo models, preserve pretraining checkpoint if it exists
                        pretraining_checkpoint = os.path.join(model_dir, 'pretraining_checkpoint')
                        has_pretraining = os.path.exists(pretraining_checkpoint)

                        if has_pretraining:
                            print(f"      Preserving pretraining checkpoint during cleanup...")

                            # Delete specific files/directories, but preserve pretraining_checkpoint
                            items_to_delete = []

                            # Add individual files to delete
                            for file in ['model.pth', 'log.json', 'hp.json']:
                                file_path = os.path.join(model_dir, file)
                                if os.path.exists(file_path):
                                    items_to_delete.append(file_path)

                            # Add directories to delete (except pretraining_checkpoint)
                            if os.path.exists(model_dir):
                                for item in os.listdir(model_dir):
                                    if item != 'pretraining_checkpoint':
                                        item_path = os.path.join(model_dir, item)
                                        if os.path.isdir(item_path):
                                            items_to_delete.append(item_path)

                            # Delete the items
                            for item_path in items_to_delete:
                                try:
                                    if os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                                    else:
                                        os.remove(item_path)
                                except Exception as e:
                                    print(f"      Failed to delete {item_path}: {e}")

                            print(f"      Cleaned up failed training (preserved pretraining checkpoint)")
                        else:
                            # No pretraining checkpoint to preserve, delete everything
                            if os.path.exists(model_dir):
                                shutil.rmtree(model_dir)
                                print(f"      Cleaned up failed training (no pretraining to preserve)")
                    else:
                        # For non-piezo models, clean up everything
                        if os.path.exists(model_dir):
                            shutil.rmtree(model_dir)
                            print(f"      Cleaned up failed training")

                    # Continue to next attempt
                    continue

            except Exception as e:
                print(f"      Training attempt {attempt} crashed: {e}")

                # Clean up on crash
                if os.path.exists(model_dir):
                    try:
                        shutil.rmtree(model_dir)
                    except:
                        pass

                # Continue to next attempt
                continue

    # NEW ENHANCEMENT 2: Convergence timing analysis (IDENTICAL to original)
    def analyze_convergence_timing(self, training_results):
        """Analyze how long networks take to reach performance thresholds."""
        print(f"\nAnalyzing Convergence Timing")
        print("=" * 60)

        convergence_results = {}

        for network_type, network_results in training_results.items():
            print(f"\nAnalyzing {network_type} convergence...")

            convergence_data = {
                'successful_runs': 0,
                'avg_training_time': 0,
                'avg_training_steps': 0,
                'convergence_stats': {},
                'raw_convergence_data': [],  # Store detailed data for report
                'individual_run_data': []  # Store per-run data
            }

            # Initialize convergence tracking
            for threshold in self.convergence_thresholds:
                convergence_data['convergence_stats'][threshold] = {
                    'times': [],
                    'steps': [],
                    'achieved_count': 0
                }

            successful_runs = []
            total_time = 0
            total_steps = 0

            for run_result in network_results['runs']:
                if run_result['status'] != 'OK' or run_result['trainer'] is None:
                    continue

                model_dir = run_result['model_dir']

                # Load performance data from all checkpoints
                performance_data = self._load_performance_data(model_dir)

                if not performance_data:
                    print(f"   No performance data found for run {run_result['run_idx']}")
                    continue

                print(f"   Found {len(performance_data)} performance checkpoints")

                successful_runs.append(run_result)
                total_time += run_result['training_time']

                # Extract performance metrics over time
                times = []
                accuracies = []
                success_probs = []
                steps = []

                for i, checkpoint in enumerate(performance_data):
                    data = checkpoint['data']

                    # Extract metrics (MODIFIED for production task metrics)
                    if 'success_action_prob' in data and 'mean_rel_action_time' in data:
                        # Estimate training time (use index as proxy if not available)
                        training_time = data.get('training_time', i * 100)  # Fallback
                        times.append(training_time)

                        # For production task, use timing accuracy instead of choice error
                        accuracy = 1 - min(data['mean_rel_action_time'], 1.0)  # Cap at 1.0
                        accuracies.append(accuracy)
                        success_probs.append(data['success_action_prob'])

                        # Estimate steps (use index as proxy)
                        step_estimate = i * 200  # Rough estimate
                        steps.append(step_estimate)

                if not times:
                    print(f"   No valid performance data for run {run_result['run_idx']}")
                    continue

                # Store individual run data for detailed analysis
                run_convergence_data = {
                    'run_idx': run_result['run_idx'],
                    'times': times,
                    'accuracies': accuracies,
                    'success_probs': success_probs,
                    'steps': steps,
                    'final_accuracy': accuracies[-1] if accuracies else 0,
                    'final_success_prob': success_probs[-1] if success_probs else 0
                }
                convergence_data['individual_run_data'].append(run_convergence_data)

                total_steps += len(steps)

                # Find convergence points for each threshold
                for threshold_name, threshold_value in self.convergence_thresholds.items():
                    if 'accuracy' in threshold_name:
                        condition = np.array(accuracies) >= threshold_value
                    else:  # valid_response
                        condition = np.array(success_probs) >= threshold_value

                    if np.any(condition):
                        first_idx = np.argmax(condition)
                        conv_time = times[first_idx]
                        conv_steps = steps[first_idx]

                        convergence_data['convergence_stats'][threshold_name]['times'].append(conv_time)
                        convergence_data['convergence_stats'][threshold_name]['steps'].append(conv_steps)
                        convergence_data['convergence_stats'][threshold_name]['achieved_count'] += 1

            # Calculate averages
            num_successful = len(successful_runs)
            if num_successful > 0:
                convergence_data['successful_runs'] = num_successful
                convergence_data['avg_training_time'] = total_time / num_successful
                convergence_data['avg_training_steps'] = total_steps / num_successful

                print(f"   Successful runs: {num_successful}")
                print(f"   Avg training time: {convergence_data['avg_training_time']:.1f}s")
                print(f"   Avg training steps: {convergence_data['avg_training_steps']:.0f}")

                # Print convergence statistics
                for threshold_name, stats in convergence_data['convergence_stats'].items():
                    achieved_count = stats['achieved_count']
                    if achieved_count > 0:
                        success_rate = achieved_count / num_successful
                        avg_time = np.mean(stats['times'])
                        avg_steps = np.mean(stats['steps'])

                        print(f"   {threshold_name}: {success_rate * 100:.1f}% achieved, "
                              f"avg {avg_time:.1f}s, {avg_steps:.0f} steps")
                    else:
                        print(f"   {threshold_name}: 0% achieved")

            convergence_results[network_type] = convergence_data

        # Save convergence analysis
        with open(os.path.join(self.output_dir, 'enhanced_convergence_analysis.pkl'), 'wb') as f:
            pickle.dump(convergence_results, f)

        return convergence_results

    def analyze_eigenvalues(self, training_results):
        """Analyze eigenvalues and spectral radii showing individual runs."""
        print(f"\nAnalyzing Network Eigenvalues & Spectral Radii - Individual Runs Analysis")
        print("=" * 60)

        eigenvalue_results = {}

        for network_type, network_results in training_results.items():
            print(f"\nAnalyzing {network_type} networks ({network_results['successful_runs']} successful runs)...")

            successful_runs = []
            eigenvalues_per_run = []
            spectral_radii = []
            eigenvalue_statistics = []

            for run_result in network_results['runs']:
                if run_result['status'] != 'OK' or run_result['trainer'] is None:
                    print(f"   Skipping run {run_result['run_idx']} - training failed")
                    continue

                model_dir = run_result['model_dir']
                hp = run_result['hp']

                if hp is None:
                    print(f"   Skipping run {run_result['run_idx']} - no hyperparameters")
                    continue

                try:
                    # Create and load model (MODIFIED: Use interval_production)
                    model = network.RNN(hp, is_cuda=False, rule_name='interval_production')

                    # Try to load the model
                    if not model.load(model_dir):
                        # Try to load from finalResult subdirectory
                        final_result_dir = os.path.join(model_dir, 'finalResult')
                        if os.path.exists(final_result_dir) and not model.load(final_result_dir):
                            print(f"   Could not load run {run_result['run_idx']} model")
                            continue

                    # Extract recurrent weight matrix
                    weight_hh = model.weight_hh.detach().cpu().numpy()

                    # Compute eigenvalues
                    eigenvalues = np.linalg.eigvals(weight_hh)
                    spectral_radius = np.max(np.abs(eigenvalues))

                    eigenvalues_per_run.append(eigenvalues)
                    spectral_radii.append(spectral_radius)
                    successful_runs.append(run_result['run_idx'])

                    # Calculate min/max statistics for real and imaginary parts
                    real_parts = eigenvalues.real
                    imag_parts = eigenvalues.imag

                    run_stats = {
                        'run_idx': run_result['run_idx'],
                        'spectral_radius': spectral_radius,
                        'real_min': np.min(real_parts),
                        'real_max': np.max(real_parts),
                        'imag_min': np.min(imag_parts),
                        'imag_max': np.max(imag_parts),
                        'num_eigenvalues': len(eigenvalues),
                        'eigenvalues_outside_unit_circle': np.sum(np.abs(eigenvalues) > 1),
                        'mean_eigenvalue_magnitude': np.mean(np.abs(eigenvalues))
                    }
                    eigenvalue_statistics.append(run_stats)

                    print(f"   Run {run_result['run_idx']}: spectral radius: {spectral_radius:.4f}")
                    print(f"      Real range: [{np.min(real_parts):.3f}, {np.max(real_parts):.3f}]")
                    print(f"      Imag range: [{np.min(imag_parts):.3f}, {np.max(imag_parts):.3f}]")
                    print(f"      Outside unit circle: {np.sum(np.abs(eigenvalues) > 1)}/{len(eigenvalues)}")

                except Exception as e:
                    print(f"   Failed to analyze run {run_result['run_idx']}: {e}")
                    continue

            # Store results with individual run data
            if len(spectral_radii) > 0:
                eigenvalue_results[network_type] = {
                    'successful_runs': successful_runs,
                    'eigenvalues_per_run': eigenvalues_per_run,
                    'spectral_radii': spectral_radii,
                    'eigenvalue_statistics': eigenvalue_statistics,
                    'mean_spectral_radius': np.mean(spectral_radii),
                    'std_spectral_radius': np.std(spectral_radii),
                    'min_spectral_radius': np.min(spectral_radii),
                    'max_spectral_radius': np.max(spectral_radii)
                }

                print(f"   {network_type} summary:")
                print(f"      Successful runs: {len(successful_runs)}")
                print(f"      Spectral radius: {np.mean(spectral_radii):.4f} ± {np.std(spectral_radii):.4f}")
                print(f"      Spectral radius range: [{np.min(spectral_radii):.4f}, {np.max(spectral_radii):.4f}]")
            else:
                print(f"   No successful runs to analyze for {network_type}")

        # Create visualization with individual runs
        if eigenvalue_results:
            self._plot_individual_runs_eigenvalues(eigenvalue_results)
            self._print_detailed_statistics(eigenvalue_results)

            # Save eigenvalue data
            with open(os.path.join(self.output_dir, 'enhanced_individual_runs_eigenvalue_analysis.pkl'), 'wb') as f:
                pickle.dump(eigenvalue_results, f)

            print(f"\nIndividual runs eigenvalue & spectral radius analysis complete!")
            print(f"   Analysis: Individual runs shown separately with detailed statistics")
        else:
            print(f"\nNo successful networks to analyze eigenvalues")

        return eigenvalue_results

    def _plot_individual_runs_eigenvalues(self, eigenvalue_results):
        """Plot eigenvalue and spectral radius analysis showing individual runs."""
        if not eigenvalue_results:
            print("No eigenvalue results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Colors for different runs
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # Plot 1: Individual eigenvalues for each run
        ax1 = axes[0, 0]

        for network_type, data in eigenvalue_results.items():
            if 'eigenvalues_per_run' not in data:
                print(f"Skipping {network_type} - no eigenvalue data")
                continue

            eigenvalues_list = data['eigenvalues_per_run']
            successful_runs = data['successful_runs']

            # Choose marker style
            marker = 'D' if network_type == 'piezo' else 'o'
            marker_size = 25 if network_type == 'piezo' else 30

            for i, (eigenvalues, run_idx) in enumerate(zip(eigenvalues_list, successful_runs)):
                color = colors[i % len(colors)]
                real_parts = eigenvalues.real
                imag_parts = eigenvalues.imag

                # Create label for first point only
                if network_type == 'piezo':
                    label = f'Piezo Run {run_idx}' if i < len(eigenvalues_list) else None
                else:
                    label = f'No-Piezo Run {run_idx}' if i < len(eigenvalues_list) else None

                ax1.scatter(real_parts, imag_parts, alpha=0.6, s=marker_size, c=color,
                            marker=marker, label=label, edgecolors='black', linewidth=0.5)

        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Individual Run Eigenvalues (Diamonds=Piezo, Circles=No-Piezo)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)

        # Plot 2: Individual spectral radii
        ax2 = axes[0, 1]

        all_spectral_radii = []
        all_labels = []
        all_colors = []

        for network_type, data in eigenvalue_results.items():
            if 'spectral_radii' not in data:
                continue

            spectral_radii = data['spectral_radii']
            successful_runs = data['successful_runs']

            for i, (sr, run_idx) in enumerate(zip(spectral_radii, successful_runs)):
                color = colors[i % len(colors)]
                if network_type == 'piezo':
                    label = f'Piezo Run {run_idx}'
                else:
                    label = f'No-Piezo Run {run_idx}'

                all_spectral_radii.append(sr)
                all_labels.append(label)
                all_colors.append(color)

        # Create bar plot with individual runs
        x_positions = np.arange(len(all_spectral_radii))
        bars = ax2.bar(x_positions, all_spectral_radii, color=all_colors, alpha=0.7)

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(all_labels, rotation=45, ha='right')
        ax2.set_ylabel('Spectral Radius')
        ax2.set_title('Individual Run Spectral Radii')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, sr in zip(bars, all_spectral_radii):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{sr:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 3: Eigenvalue magnitude distribution by run
        ax3 = axes[1, 0]

        for network_type, data in eigenvalue_results.items():
            if 'eigenvalues_per_run' not in data:
                continue

            eigenvalues_list = data['eigenvalues_per_run']
            successful_runs = data['successful_runs']

            for i, (eigenvalues, run_idx) in enumerate(zip(eigenvalues_list, successful_runs)):
                color = colors[i % len(colors)]
                magnitudes = np.abs(eigenvalues)

                if network_type == 'piezo':
                    label = f'Piezo Run {run_idx}'
                    linestyle = '--'
                else:
                    label = f'No-Piezo Run {run_idx}'
                    linestyle = '-'

                ax3.hist(magnitudes, bins=30, alpha=0.5, color=color, label=label,
                         density=True, histtype='step', linewidth=2, linestyle=linestyle)

        ax3.set_xlabel('Eigenvalue Magnitude')
        ax3.set_ylabel('Density')
        ax3.set_title('Individual Run Eigenvalue Magnitude Distributions')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics comparison
        ax4 = axes[1, 1]

        # Create summary plot comparing network types
        network_types = list(eigenvalue_results.keys())
        if len(network_types) >= 2:
            # Compare statistics between network types
            stats_comparison = {}

            for network_type, data in eigenvalue_results.items():
                if 'spectral_radii' in data:
                    stats_comparison[network_type] = {
                        'mean_sr': np.mean(data['spectral_radii']),
                        'std_sr': np.std(data['spectral_radii']),
                        'min_sr': np.min(data['spectral_radii']),
                        'max_sr': np.max(data['spectral_radii'])
                    }

            # Plot means with error bars
            types = list(stats_comparison.keys())
            means = [stats_comparison[t]['mean_sr'] for t in types]
            stds = [stats_comparison[t]['std_sr'] for t in types]

            # Make readable labels
            readable_labels = []
            for t in types:
                if t == 'no_piezo':
                    readable_labels.append('No Piezo')
                elif t == 'piezo':
                    readable_labels.append('With Piezo')
                else:
                    readable_labels.append(t.replace('_', ' ').title())

            bars = ax4.bar(readable_labels, means, yerr=stds,
                           color=['blue', 'red'][:len(types)], alpha=0.7, capsize=8)

            # Overlay individual points
            for i, network_type in enumerate(types):
                data = eigenvalue_results[network_type]
                spectral_radii = data['spectral_radii']
                jittered_x = np.random.normal(i, 0.05, len(spectral_radii))
                ax4.scatter(jittered_x, spectral_radii, color='black', s=40, alpha=0.7, zorder=10)

            ax4.set_ylabel('Spectral Radius')
            ax4.set_title('Network Type Comparison (Mean ± Std)')
            ax4.grid(True, alpha=0.3)

            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax4.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 network types for comparison',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Network Type Comparison - Insufficient Data')

        plt.tight_layout()

        # Save plot
        delay_suffix = "_with_delay" if self.use_time_delay else "_no_delay"
        plt.savefig(os.path.join(self.output_dir, f'enhanced_individual_runs_eigenvalue_analysis{delay_suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Individual runs eigenvalue & spectral radius visualization saved")

    def _print_detailed_statistics(self, eigenvalue_results):
        """Print detailed statistics for each run."""
        print(f"\nDETAILED EIGENVALUE STATISTICS PER RUN")
        print("=" * 60)

        for network_type, data in eigenvalue_results.items():
            if 'eigenvalue_statistics' not in data:
                continue

            print(f"\n{network_type.upper()} NETWORKS:")
            print("-" * 40)

            stats_list = data['eigenvalue_statistics']

            # Print individual run statistics
            for stats in stats_list:
                print(f"  Run {stats['run_idx']}:")
                print(f"    Spectral Radius: {stats['spectral_radius']:.4f}")
                print(f"    Real part range: [{stats['real_min']:.3f}, {stats['real_max']:.3f}]")
                print(f"    Imag part range: [{stats['imag_min']:.3f}, {stats['imag_max']:.3f}]")
                print(f"    Outside unit circle: {stats['eigenvalues_outside_unit_circle']}/{stats['num_eigenvalues']}")
                print(f"    Mean eigenvalue magnitude: {stats['mean_eigenvalue_magnitude']:.3f}")
                print()

            # Calculate aggregate statistics across runs
            if len(stats_list) > 0:
                real_mins = [s['real_min'] for s in stats_list]
                real_maxs = [s['real_max'] for s in stats_list]
                imag_mins = [s['imag_min'] for s in stats_list]
                imag_maxs = [s['imag_max'] for s in stats_list]

                print(f"  AGGREGATE STATISTICS ACROSS {len(stats_list)} RUNS:")
                print(f"    Real part absolute range: [{min(real_mins):.3f}, {max(real_maxs):.3f}]")
                print(f"    Imag part absolute range: [{min(imag_mins):.3f}, {max(imag_maxs):.3f}]")
                print(f"    Real part range variation: {np.std(np.array(real_maxs) - np.array(real_mins)):.3f}")
                print(f"    Imag part range variation: {np.std(np.array(imag_maxs) - np.array(imag_mins)):.3f}")
                print()

    def simulate_piezo_response(self):
        """Simulate piezo response for task-relevant durations."""
        print(f"\nSimulating Task-Relevant Piezo Response")
        print("=" * 60)

        # TASK-RELEVANT DURATION: Match actual timing task length
        task_duration_seconds = 6.5  # Slightly longer for safety
        sequence_length = int(task_duration_seconds * 60)  # Convert to samples at 60 Hz

        print(f"Task-relevant simulation:")
        print(f"   Duration: {task_duration_seconds} seconds (matches timing task)")
        print(f"   Samples: {sequence_length} at 60 Hz")
        print(f"   Enhanced intervals: {self.enhanced_intervals['min_interval']}-{self.enhanced_intervals['max_interval']}ms")

        # Generate realistic cardiac pressure for task duration
        pressure_data = generate_simple_cardiac_pressure(sequence_length, sampling_rate=60)

        # Create realistic heartbeat sequence
        hb_sequence, r_peaks = create_sparse_hb_sequence(pressure_data, sampling_rate=60)

        # Calculate expected vs actual R-peaks
        expected_r_peaks = task_duration_seconds / 0.8  # 0.8 seconds per beat at 75 BPM
        actual_r_peaks = len(r_peaks)

        print(f"   Expected R-peaks: {expected_r_peaks:.1f}")
        print(f"   Actual R-peaks: {actual_r_peaks}")
        print(f"   Heart rate: {actual_r_peaks / task_duration_seconds * 60:.0f} BPM")

        # Create piezo interface with connectivity verification
        print(f"\nCreating and verifying piezo interface...")
        piezo_interface = SimplePiezoInterface(
            num_neurons=256,
            connection_fraction=0.15,
            slice_size=20,
            use_temporal_delay=self.use_time_delay,
            delay_steps=3
        )

        # Simulate slicing and piezo response
        slice_size = 20
        num_slices = len(pressure_data) // slice_size

        slice_means = []
        piezo_responses = []
        slice_times = []
        min_pressure = np.min(pressure_data)

        print(f"Processing {num_slices} slices ({num_slices * slice_size / 60:.1f} seconds)...")

        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            # Calculate slice time
            slice_time = (start_idx + slice_size // 2) / 60  # Middle of slice in seconds
            slice_times.append(slice_time)

            # Extract slice
            if end_idx <= len(pressure_data):
                slice_data = pressure_data[start_idx:end_idx]
            else:
                slice_data = pressure_data[start_idx:]
                if len(slice_data) < slice_size:
                    slice_data = np.pad(slice_data, (0, slice_size - len(slice_data)), 'constant')

            # Calculate mean pressure of slice
            slice_mean = np.mean(slice_data)
            slice_means.append(slice_mean)

            # Convert to tensor and get piezo response
            slice_tensor = torch.tensor(slice_data, dtype=torch.float32)
            min_pressure_tensor = torch.tensor(min_pressure, dtype=torch.float32)

            # Get piezo response
            response = piezo_interface(slice_tensor, min_pressure_tensor)
            mean_response = torch.mean(response[piezo_interface.connected_indices]).item()
            piezo_responses.append(mean_response)

        # Create task-relevant visualization
        self._plot_piezo_response(slice_times, slice_means, piezo_responses,
                                  r_peaks, slice_size, task_duration_seconds,
                                  actual_r_peaks, pressure_data)

        # Calculate task-relevant statistics
        correlation = np.corrcoef(slice_means, piezo_responses)[0, 1]
        heart_rate = actual_r_peaks / task_duration_seconds * 60
        cardiac_cycles_per_interval = (self.enhanced_intervals['max_interval'] / 1000) / 0.8

        print(f"\nTask-relevant piezo simulation complete!")
        print(f"   Processed {len(slice_means)} slices")
        print(f"   Heart rate: {heart_rate:.0f} BPM ({actual_r_peaks} R-peaks)")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Cardiac cycles per max interval: {cardiac_cycles_per_interval:.1f}")
        print(f"   This is what happens during ONE timing task!")

        # Save data
        response_data = {
            'slice_means': slice_means,
            'piezo_responses': piezo_responses,
            'slice_times': slice_times,
            'r_peaks': r_peaks,
            'slice_size': slice_size,
            'pressure_data': pressure_data,
            'task_duration': task_duration_seconds,
            'heart_rate': heart_rate,
            'correlation': correlation,
            'cardiac_cycles_per_interval': cardiac_cycles_per_interval,
            'use_time_delay': self.use_time_delay
        }

        with open(os.path.join(self.output_dir, 'enhanced_task_relevant_piezo_data.pkl'), 'wb') as f:
            pickle.dump(response_data, f)

        return response_data

    def _plot_piezo_response(self, slice_times, slice_means, piezo_responses,
                             r_peaks, slice_size, task_duration, actual_r_peaks, pressure_data):
        """Plot task-relevant piezo response analysis."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Plot 1: Full pressure signal with task timing overlay
        ax1 = axes[0]
        full_time = np.arange(len(pressure_data)) / 60
        ax1.plot(full_time, pressure_data, 'b-', linewidth=1.5, label='Cardiac Pressure')

        # Add R-peak markers
        for r_peak in r_peaks:
            r_peak_time = r_peak / 60
            ax1.axvline(r_peak_time, color='r', linestyle='--', alpha=0.7, linewidth=1)

        # Add task timing overlay
        max_interval_sec = self.enhanced_intervals['max_interval'] / 1000
        ax1.axvspan(0, max_interval_sec, alpha=0.2, color='green', label=f'Max Interval ({max_interval_sec}s)')

        ax1.set_ylabel('Pressure')
        ax1.set_title(f'Task-Relevant Cardiac Pressure ({task_duration:.1f}s = 1 Timing Task)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, task_duration)

        # Plot 2: Slice means with task phases
        ax2 = axes[1]
        ax2.plot(slice_times, slice_means, 'b-', linewidth=2, label='Slice Mean Pressure')

        # Show task phases for production task
        interval_end = max_interval_sec
        delay_end = interval_end + 1.0  # 1 second delay
        production_end = delay_end + max_interval_sec

        ax2.axvspan(0, interval_end, alpha=0.2, color='green', label='Initial Interval')
        ax2.axvspan(interval_end, delay_end, alpha=0.2, color='yellow', label='Delay')
        ax2.axvspan(delay_end, production_end, alpha=0.2, color='orange', label='Production')

        ax2.set_ylabel('Mean Pressure')
        ax2.set_title('Cardiac Pressure During Production Task Phases')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, task_duration)

        # Plot 3: Piezo responses with task phases
        ax3 = axes[2]
        ax3.plot(slice_times, piezo_responses, 'g-', linewidth=2, label='Piezo Response')

        # Same task phases
        ax3.axvspan(0, interval_end, alpha=0.2, color='green', label='Initial Interval')
        ax3.axvspan(interval_end, delay_end, alpha=0.2, color='yellow', label='Delay')
        ax3.axvspan(delay_end, production_end, alpha=0.2, color='orange', label='Production')

        ax3.set_ylabel('Piezo Response')
        ax3.set_title(f'Piezo Response During Production Task (Time Delay: {self.use_time_delay})')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim(0, task_duration)

        # Plot 4: Task-relevant statistics
        ax4 = axes[3]
        ax4.scatter(slice_means, piezo_responses, alpha=0.7, s=30, c='purple')
        ax4.set_xlabel('Slice Mean Pressure')
        ax4.set_ylabel('Piezo Response')
        ax4.set_title('Task-Relevant Piezo Response vs Pressure')
        ax4.grid(True, alpha=0.3)

        # Add task-relevant statistics
        corr = np.corrcoef(slice_means, piezo_responses)[0, 1]
        heart_rate = actual_r_peaks / task_duration * 60

        stats_text = f'Correlation: {corr:.3f}\n'
        stats_text += f'Heart Rate: {heart_rate:.0f} BPM\n'
        stats_text += f'R-peaks per task: {actual_r_peaks}\n'
        stats_text += f'Task duration: {task_duration:.1f}s'

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top')

        plt.tight_layout()

        # Save plot
        delay_suffix = "_with_delay" if self.use_time_delay else "_no_delay"
        plt.savefig(os.path.join(self.output_dir, f'enhanced_task_relevant_piezo{delay_suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Task-relevant visualization saved")

    def _load_performance_data(self, model_dir):
        """Load performance data from all available checkpoint directories."""
        performance_data = []

        # Check numbered checkpoints (0, 1, 2, ...)
        checkpoint_idx = 0
        while True:
            checkpoint_dir = os.path.join(model_dir, str(checkpoint_idx))
            if not os.path.exists(checkpoint_dir):
                break

            checkpoint_log = tools.load_log(checkpoint_dir)
            if checkpoint_log:
                performance_data.append({
                    'checkpoint': str(checkpoint_idx),
                    'data': checkpoint_log,
                    'type': 'numbered'
                })
            checkpoint_idx += 1

        # Check finalResult directory
        final_result_dir = os.path.join(model_dir, 'finalResult')
        if os.path.exists(final_result_dir):
            final_log = tools.load_log(final_result_dir)
            if final_log:
                performance_data.append({
                    'checkpoint': 'finalResult',
                    'data': final_log,
                    'type': 'final'
                })

        return performance_data

    def _calculate_mean_accuracies(self, training_results):
        """Calculate mean final accuracies for each network type."""
        print(f"\nCalculating Mean Accuracies")
        print("-" * 30)

        accuracy_stats = {}

        for network_type, network_results in training_results.items():
            final_accuracies = []
            final_success_probs = []
            final_timing_errors = []  # MODIFIED for production task

            print(f"\nAnalyzing {network_type} final accuracies...")

            for run_result in network_results['runs']:
                if run_result['status'] != 'OK':
                    continue

                # Load final performance data
                performance_data = self._load_performance_data(run_result['model_dir'])

                if performance_data:
                    # Use the last checkpoint (preferring finalResult if available)
                    final_data = None

                    # Look for finalResult first
                    for checkpoint in performance_data:
                        if checkpoint['type'] == 'final':
                            final_data = checkpoint['data']
                            break

                    # If no finalResult, use the last numbered checkpoint
                    if final_data is None and performance_data:
                        final_data = performance_data[-1]['data']

                    # MODIFIED for production task metrics
                    if final_data and 'mean_rel_action_time' in final_data:
                        timing_error = final_data['mean_rel_action_time']
                        accuracy = 1 - min(timing_error, 1.0)  # Cap at 1.0
                        final_accuracies.append(accuracy)
                        final_timing_errors.append(timing_error)
                        print(f"   Run {run_result['run_idx']}: accuracy = {accuracy:.3f}, timing error = {timing_error:.3f}")

                    if final_data and 'success_action_prob' in final_data:
                        final_success_probs.append(final_data['success_action_prob'])

            if final_accuracies:
                accuracy_stats[network_type] = {
                    'mean_accuracy': np.mean(final_accuracies),
                    'std_accuracy': np.std(final_accuracies),
                    'min_accuracy': np.min(final_accuracies),
                    'max_accuracy': np.max(final_accuracies),
                    'mean_timing_error': np.mean(final_timing_errors),  # MODIFIED
                    'std_timing_error': np.std(final_timing_errors),    # MODIFIED
                    'mean_success_prob': np.mean(final_success_probs) if final_success_probs else 0,
                    'std_success_prob': np.std(final_success_probs) if final_success_probs else 0,
                    'num_runs': len(final_accuracies),
                    'individual_accuracies': final_accuracies,
                    'individual_success_probs': final_success_probs
                }

                stats = accuracy_stats[network_type]
                print(f"   Mean accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
                print(f"   Range: [{stats['min_accuracy']:.3f}, {stats['max_accuracy']:.3f}]")
                print(f"   Mean timing error: {stats['mean_timing_error']:.3f}")
                print(f"   Mean success prob: {stats['mean_success_prob']:.3f}")
                print(f"   Based on {stats['num_runs']} runs")

        return accuracy_stats

    def _plot_individual_model_performance(self, training_results):
        """Generate individual performance graphs for each model run."""
        print(f"\nGenerating Individual Model Performance Graphs")
        print("-" * 45)

        plots_created = 0

        for network_type, network_results in training_results.items():
            print(f"\nCreating plots for {network_type} models...")

            for run_result in network_results['runs']:
                if run_result['status'] != 'OK':
                    print(f"   Skipping failed run {run_result['run_idx']}")
                    continue

                model_dir = run_result['model_dir']
                run_idx = run_result['run_idx']

                # Load performance data for this specific model
                performance_data = self._load_performance_data(model_dir)

                if not performance_data:
                    print(f"   No performance data for run {run_idx}")
                    continue

                # Extract metrics (MODIFIED for production task)
                times = []
                accuracies = []
                success_probs = []
                timing_errors = []  # MODIFIED
                checkpoint_labels = []

                for checkpoint in performance_data:
                    data = checkpoint['data']
                    checkpoint_name = checkpoint['checkpoint']

                    if 'success_action_prob' in data and 'mean_rel_action_time' in data:
                        # Use training time if available, otherwise estimate
                        training_time = data.get('training_time', len(times) * 100)
                        times.append(training_time)

                        timing_error = data['mean_rel_action_time']
                        accuracy = 1 - min(timing_error, 1.0)  # Cap at 1.0
                        accuracies.append(accuracy)
                        success_probs.append(data['success_action_prob'])
                        timing_errors.append(timing_error)
                        checkpoint_labels.append(checkpoint_name)

                if not times:
                    print(f"   No valid performance data for run {run_idx}")
                    continue

                # Create individual plot (MODIFIED for production task)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

                # Plot 1: Accuracy over time
                ax1.plot(times, accuracies, 'b-', marker='o', linewidth=2, markersize=6)
                ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% threshold')
                ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
                ax1.set_ylabel('Accuracy')
                ax1.set_title(f'{network_type.title()} Run {run_idx} - Accuracy Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_ylim(0, 1.05)

                # Plot 2: Success probability over time
                ax2.plot(times, success_probs, 'r-', marker='s', linewidth=2, markersize=6)
                ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% threshold')
                ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
                ax2.set_ylabel('Success Probability')
                ax2.set_title(f'{network_type.title()} Run {run_idx} - Success Probability Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.set_ylim(0, 1.05)

                # Plot 3: Timing error over time (MODIFIED for production task)
                ax3.plot(times, timing_errors, 'orange', marker='^', linewidth=2, markersize=6)
                ax3.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='10% error')
                ax3.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% error')
                ax3.set_xlabel('Training Time (s)')
                ax3.set_ylabel('Relative Timing Error')
                ax3.set_title(f'{network_type.title()} Run {run_idx} - Timing Error Over Time')
                ax3.grid(True, alpha=0.3)
                ax3.legend()

                # Plot 4: Final performance summary
                ax4.bar(['Accuracy', 'Success Prob'],
                        [accuracies[-1], success_probs[-1]],
                        color=['blue', 'red'], alpha=0.7)
                ax4.set_ylabel('Final Performance')
                ax4.set_title(f'{network_type.title()} Run {run_idx} - Final Performance')
                ax4.set_ylim(0, 1.05)
                ax4.grid(True, alpha=0.3)

                # Add values on bars
                ax4.text(0, accuracies[-1] + 0.02, f'{accuracies[-1]:.3f}',
                         ha='center', fontweight='bold')
                ax4.text(1, success_probs[-1] + 0.02, f'{success_probs[-1]:.3f}',
                         ha='center', fontweight='bold')

                plt.suptitle(f'{network_type.title()} Model Run {run_idx} Performance Analysis (Production Task)',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()

                # Save individual plot
                delay_suffix = "_with_delay" if self.use_time_delay else "_no_delay"
                plot_filename = f'individual_{network_type}_run_{run_idx}_performance{delay_suffix}.png'
                plt.savefig(os.path.join(self.output_dir, plot_filename), dpi=300, bbox_inches='tight')
                plt.close()

                print(f"   Created plot for run {run_idx}: {plot_filename}")
                plots_created += 1

        print(f"\nTotal individual plots created: {plots_created}")

    def _save_multiple_runs_summary(self, results):
        """Save training summary for multiple runs."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_intervals': self.enhanced_intervals,
            'use_time_delay': self.use_time_delay,
            'num_runs': self.num_runs,
            'networks': {}
        }

        for network_type, network_results in results.items():
            successful_runs = network_results['successful_runs']
            failed_runs = network_results['failed_runs']
            total_time = network_results['total_training_time']

            network_summary = {
                'total_runs': self.num_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / self.num_runs,
                'total_training_time': total_time,
                'average_training_time': total_time / self.num_runs if self.num_runs > 0 else 0,
                'runs_details': []
            }

            for run_result in network_results['runs']:
                run_detail = {
                    'run_idx': run_result['run_idx'],
                    'status': run_result['status'],
                    'training_time': run_result['training_time'],
                    'model_dir': run_result['model_dir'],
                    'successful': run_result['status'] == 'OK'
                }
                network_summary['runs_details'].append(run_detail)

            summary['networks'][network_type] = network_summary

        with open(os.path.join(self.output_dir, 'enhanced_consolidated_analysis_summary.json'), 'w') as f:
            import json
            json.dump(summary, f, indent=2)

        print(f"Enhanced consolidated analysis summary saved")

    def run_full_comparison(self, max_samples=5e5, load_existing=False):
        """Run the complete enhanced comparison analysis."""
        print(f"\nENHANCED PIEZO VS NON-PIEZO COMPREHENSIVE COMPARISON (PRODUCTION TASK)")
        print("=" * 80)
        print(f"Analysis Parameters:")
        print(f"   Task: Interval Production")  # MODIFIED
        print(f"   Enhanced Intervals: {self.enhanced_intervals['min_interval']}-{self.enhanced_intervals['max_interval']}ms")
        print(f"   Max Samples: {max_samples:,.0f}")
        print(f"   Runs per network: {self.num_runs}")
        print(f"   Time Delay: {'ENABLED' if self.use_time_delay else 'DISABLED'}")
        print(f"   Load existing: {'ENABLED' if load_existing else 'DISABLED'}")
        print(f"   Output: {self.output_dir}")
        print("ENHANCEMENTS:")
        print("   Connectivity parameter tracking (initial vs final + correlation)")
        print("   Convergence timing analysis (time to reach 90%/95% thresholds)")

        # Step 1: Simulate piezo response (independent of training)
        print(f"\nStep 1: Piezo Response Simulation")
        piezo_data = self.simulate_piezo_response()

        # Step 2: Train or load networks
        if load_existing:
            print(f"\nStep 2: Loading Existing Networks (NOT IMPLEMENTED)")
            print("   This enhanced version focuses on training new networks")
            print("   For loading existing, use original piezo_comparison.py")
            return None
        else:
            print(f"\nStep 2: Training New Networks with Enhanced Tracking")
            training_results = self.train_networks(max_samples)

        # Step 3: Enhanced convergence analysis
        print(f"\nStep 3: Enhanced Convergence Analysis")
        convergence_results = self.analyze_convergence_timing(training_results)

        # Step 4: Calculate mean accuracies
        print(f"\nStep 4: Mean Accuracy Analysis")
        accuracy_stats = self._calculate_mean_accuracies(training_results)

        # Step 5: Generate individual model graphs
        print(f"\nStep 5: Individual Model Visualizations")
        self._plot_individual_model_performance(training_results)

        # Step 6: Analyze eigenvalues and spectral radii
        print(f"\nStep 6: Individual Runs Eigenvalue & Spectral Radius Analysis")
        eigenvalue_results = self.analyze_eigenvalues(training_results)

        # Step 7: Generate final enhanced report
        print(f"\nStep 7: Enhanced Final Report")
        self._generate_enhanced_report(training_results, eigenvalue_results, piezo_data, convergence_results, accuracy_stats)

        print(f"\nENHANCED COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"All enhanced results saved to: {self.output_dir}")

        return {
            'training_results': training_results,
            'eigenvalue_results': eigenvalue_results,
            'piezo_data': piezo_data,
            'convergence_results': convergence_results
        }

    def _generate_enhanced_report(self, training_results, eigenvalue_results, piezo_data,
                                  convergence_results, accuracy_stats):
        """Generate enhanced final report with connectivity and convergence analysis."""
        report = []
        report.append("ENHANCED PIEZO VS NON-PIEZO NETWORK COMPARISON (PRODUCTION TASK)")
        report.append("=" * 70)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Training runs per network: {self.num_runs}")
        report.append(f"Time Delay: {'ENABLED' if self.use_time_delay else 'DISABLED'}")
        report.append(f"Max samples per run: {5e5:,.0f}")
        report.append("")
        report.append("TASK: INTERVAL PRODUCTION")
        report.append("Networks learn to reproduce time intervals")
        report.append("See two pulses -> learn to output after same interval")
        report.append("")
        report.append("ENHANCEMENTS:")
        report.append("   Connectivity parameter tracking (initial vs final + correlation)")
        report.append("   Convergence timing analysis")
        report.append("   Individual model performance graphs")
        report.append("   Mean accuracy analysis")
        report.append("   Enhanced statistics and visualizations")
        report.append("")

        # Training Results Summary
        report.append("MULTIPLE-RUN TRAINING RESULTS")
        report.append("-" * 30)

        total_successful = 0
        total_failed = 0

        for network_type, network_results in training_results.items():
            successful = network_results['successful_runs']
            failed = network_results['failed_runs']
            total_time = network_results['total_training_time']
            success_rate = successful / self.num_runs * 100

            total_successful += successful
            total_failed += failed

            report.append(f"{network_type.upper()}:")
            report.append(f"  Successful runs: {successful}/{self.num_runs} ({success_rate:.1f}%)")
            report.append(f"  Failed runs: {failed}/{self.num_runs}")
            report.append(f"  Total training time: {total_time:.1f}s")
            report.append(f"  Average time per run: {total_time / self.num_runs:.1f}s")
            report.append("")

        report.append(f"OVERALL TRAINING SUMMARY:")
        report.append(f"  Total successful runs: {total_successful}/{self.num_runs * 2}")
        report.append(f"  Total failed runs: {total_failed}/{self.num_runs * 2}")
        report.append(f"  Overall success rate: {total_successful / (self.num_runs * 2) * 100:.1f}%")
        report.append("")

        # Enhanced Connectivity Analysis
        report.append("ENHANCED CONNECTIVITY ANALYSIS (PIEZO ONLY)")
        report.append("-" * 45)

        connectivity_stats = []
        if 'piezo' in training_results:
            for run_result in training_results['piezo']['runs']:
                if run_result['status'] == 'OK':
                    connectivity_file = os.path.join(run_result['model_dir'], 'connectivity_analysis.pkl')
                    if os.path.exists(connectivity_file):
                        try:
                            with open(connectivity_file, 'rb') as f:
                                conn_data = pickle.load(f)
                            connectivity_stats.append(conn_data)
                        except:
                            continue

        if connectivity_stats:
            raw_correlations = []
            activated_correlations = []
            weight_changes = []
            initial_means = []
            final_means = []

            for conn_data in connectivity_stats:
                if conn_data.get('correlation'):
                    corr = conn_data['correlation']
                    raw_correlations.append(corr['raw_correlation'])
                    activated_correlations.append(corr['activated_correlation'])
                    weight_changes.append(corr['max_activated_change'])

                if conn_data.get('initial') and conn_data.get('final'):
                    initial = conn_data['initial']
                    final = conn_data['final']
                    initial_means.append(initial['activated_mean'])
                    final_means.append(final['activated_mean'])

            if raw_correlations:
                report.append(f"Connectivity Parameter Evolution:")
                report.append(f"  Raw weight correlation: {np.mean(raw_correlations):.3f} ± {np.std(raw_correlations):.3f}")
                report.append(f"  Activated weight correlation: {np.mean(activated_correlations):.3f} ± {np.std(activated_correlations):.3f}")
                report.append(f"  Max weight change: {np.mean(weight_changes):.3f} ± {np.std(weight_changes):.3f}")

            if initial_means and final_means:
                report.append(f"  Initial connectivity mean: {np.mean(initial_means):.3f} ± {np.std(initial_means):.3f}")
                report.append(f"  Final connectivity mean: {np.mean(final_means):.3f} ± {np.std(final_means):.3f}")
                from scipy import stats
                # Statistical test for change
                t_stat, p_value = stats.ttest_rel(final_means, initial_means)
                report.append(f"  Connectivity change: t={t_stat:.3f}, p={p_value:.4f}")
                if p_value < 0.05:
                    report.append(f"  SIGNIFICANT connectivity change during training")
                else:
                    report.append(f"  No significant connectivity change during training")

            report.append(f"  Analysis based on {len(connectivity_stats)} successful piezo runs")
        else:
            report.append("No connectivity data available (no successful piezo runs)")

        report.append("")

        # Enhanced Convergence Analysis
        report.append("ENHANCED CONVERGENCE ANALYSIS")
        report.append("-" * 32)

        for network_type, conv_data in convergence_results.items():
            if conv_data['successful_runs'] > 0:
                report.append(f"{network_type.upper()}:")
                report.append(f"  Successful runs: {conv_data['successful_runs']}")
                report.append(f"  Avg training time: {conv_data['avg_training_time']:.1f}s")
                report.append(f"  Avg training steps: {conv_data['avg_training_steps']:.0f}")

                # Report convergence to different thresholds
                for threshold_name, stats in conv_data['convergence_stats'].items():
                    achieved = stats['achieved_count']
                    total = conv_data['successful_runs']
                    if achieved > 0:
                        success_rate = achieved / total * 100
                        avg_time = np.mean(stats['times'])
                        avg_steps = np.mean(stats['steps'])
                        report.append(f"    {threshold_name}: {success_rate:.1f}% achieved, avg {avg_time:.1f}s, {avg_steps:.0f} steps")
                    else:
                        report.append(f"    {threshold_name}: 0% achieved")
                report.append("")

        # Mean Accuracy Analysis (MODIFIED for production task)
        report.append("MEAN ACCURACY ANALYSIS (PRODUCTION TASK)")
        report.append("-" * 38)

        for network_type, stats in accuracy_stats.items():
            report.append(f"{network_type.upper()}:")
            report.append(f"  Mean final accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
            report.append(f"  Accuracy range: [{stats['min_accuracy']:.3f}, {stats['max_accuracy']:.3f}]")
            report.append(f"  Mean timing error: {stats['mean_timing_error']:.3f} ± {stats['std_timing_error']:.3f}")
            report.append(f"  Mean success probability: {stats['mean_success_prob']:.3f} ± {stats['std_success_prob']:.3f}")
            report.append(f"  Based on {stats['num_runs']} successful runs")
            report.append("")

        # Eigenvalue Analysis Results
        report.append("INDIVIDUAL RUNS EIGENVALUE & SPECTRAL RADIUS ANALYSIS")
        report.append("-" * 55)

        if eigenvalue_results:
            for network_type, data in eigenvalue_results.items():
                report.append(f"{network_type.upper()}:")
                report.append(f"  Analyzed runs: {len(data['successful_runs'])}")
                report.append(f"  Spectral radius: {data['mean_spectral_radius']:.4f} ± {data['std_spectral_radius']:.4f}")
                report.append(f"  Spectral radius range: [{data['min_spectral_radius']:.4f}, {data['max_spectral_radius']:.4f}]")

                if 'eigenvalue_statistics' in data:
                    stats_list = data['eigenvalue_statistics']
                    if stats_list:
                        spectral_radii = [s['spectral_radius'] for s in stats_list]
                        unstable_runs = sum(1 for sr in spectral_radii if sr > 1.1)
                        report.append(f"  Potentially unstable runs: {unstable_runs}/{len(spectral_radii)}")
                report.append("")
        else:
            report.append("No eigenvalue analysis available (no successful trainings)")
            report.append("")

        # Piezo Response Analysis
        report.append("PIEZO RESPONSE ANALYSIS")
        report.append("-" * 20)
        correlation = np.corrcoef(piezo_data['slice_means'], piezo_data['piezo_responses'])[0, 1]
        report.append(f"Slice-Response Correlation: {correlation:.3f}")
        report.append(f"Heart Rate: {piezo_data['heart_rate']:.0f} BPM")
        report.append(f"R-Peaks per Task: {len(piezo_data['r_peaks'])}")
        report.append(f"Cardiac Cycles per Max Interval: {piezo_data['cardiac_cycles_per_interval']:.1f}")
        report.append("")

        # Enhanced Comparative Summary
        report.append("ENHANCED COMPARATIVE SUMMARY")
        report.append("-" * 30)

        # Training efficiency comparison
        if 'no_piezo' in training_results and 'piezo' in training_results:
            no_piezo_time = training_results['no_piezo']['total_training_time'] / max(training_results['no_piezo']['successful_runs'], 1)
            piezo_time = training_results['piezo']['total_training_time'] / max(training_results['piezo']['successful_runs'], 1)

            time_ratio = piezo_time / no_piezo_time if no_piezo_time > 0 else float('inf')
            report.append(f"Training Time Comparison:")
            report.append(f"  No-Piezo avg: {no_piezo_time:.1f}s")
            report.append(f"  Piezo avg: {piezo_time:.1f}s")
            report.append(f"  Piezo/No-Piezo ratio: {time_ratio:.2f}x")

            if time_ratio > 1.2:
                report.append(f"  Piezo models take {time_ratio:.1f}x longer to train")
            elif time_ratio < 0.8:
                report.append(f"  Piezo models train {1/time_ratio:.1f}x faster")
            else:
                report.append(f"  Similar training times")

        # Accuracy comparison
        if 'no_piezo' in accuracy_stats and 'piezo' in accuracy_stats:
            no_piezo_acc = accuracy_stats['no_piezo']['mean_accuracy']
            piezo_acc = accuracy_stats['piezo']['mean_accuracy']

            report.append(f"Final Accuracy Comparison:")
            report.append(f"  No-Piezo: {no_piezo_acc:.3f} ± {accuracy_stats['no_piezo']['std_accuracy']:.3f}")
            report.append(f"  Piezo: {piezo_acc:.3f} ± {accuracy_stats['piezo']['std_accuracy']:.3f}")

            acc_diff = piezo_acc - no_piezo_acc
            if abs(acc_diff) > 0.01:  # 1% difference threshold
                if acc_diff > 0:
                    report.append(f"  Piezo models achieve {acc_diff:.3f} higher accuracy")
                else:
                    report.append(f"  No-Piezo models achieve {abs(acc_diff):.3f} higher accuracy")
            else:
                report.append(f"  Similar final accuracies")

        report.append("")

        # Save enhanced report
        report_text = "\n".join(report)
        with open(os.path.join(self.output_dir, 'enhanced_analysis_report.txt'), 'w') as f:
            f.write(report_text)

        print("Enhanced analysis report saved!")

        # Print summary to console
        print(f"\nENHANCED ANALYSIS SUMMARY:")
        print(f"   Overall success rate: {total_successful / (self.num_runs * 2) * 100:.1f}%")
        print(f"   Total training runs: {self.num_runs * 2}")
        if accuracy_stats:
            for network_type, stats in accuracy_stats.items():
                print(f"   {network_type.title()} mean accuracy: {stats['mean_accuracy']:.3f}")
        print(f"   Eigenvalue analysis: {'Available' if eigenvalue_results else 'Not available'}")
        print(f"   Piezo correlation: {correlation:.3f}")
        print(f"   Connectivity analysis: {'Available' if connectivity_stats else 'Not available'}")
        print(f"   Convergence analysis: {'Available' if convergence_results else 'Not available'}")
        print(f"   Individual plots created: Available")
        print(f"   Full enhanced report: {os.path.join(self.output_dir, 'enhanced_analysis_report.txt')}")


def main():
    """Main function to run the enhanced production comparison analysis."""
    parser = argparse.ArgumentParser(description='Enhanced Piezo vs Non-Piezo Network Comparison - Production Task')
    parser.add_argument('--full-training', action='store_true',
                        help='Run full training (2M samples) instead of quick test (500k)')
    parser.add_argument('--time-delay', action='store_true',
                        help='Enable time delay in piezo interface')
    parser.add_argument('--output-dir', default='enhanced_piezo_production_results',
                        help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of training runs per network type (default: 10)')
    parser.add_argument('--load-existing', action='store_true',
                        help='Load existing trained models (not implemented in enhanced version)')

    args = parser.parse_args()

    if args.load_existing:
        print("Loading existing models is not implemented in the enhanced version.")
        print("   For loading existing models, use the original piezo_comparison.py")
        print("   This enhanced version focuses on training new networks with detailed tracking.")
        return

    # Create enhanced analyzer
    analyzer = EnhancedPiezoProductionAnalyzer(
        output_dir=args.output_dir,
        use_time_delay=args.time_delay,
        num_runs=args.num_runs
    )

    # Run enhanced comparison
    max_samples = 2e6 if args.full_training else 5e5
    analyzer.run_full_comparison(max_samples, load_existing=False)


if __name__ == "__main__":
    main()