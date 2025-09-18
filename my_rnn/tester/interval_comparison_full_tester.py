"""
Enhanced Interval Comparison Full Tester - UNIFIED VERSION
=========================================================

Tests all three types of trained interval comparison models with comprehensive analysis:
- Standard models (no piezo, no insula)
- Piezo models (piezo enabled, no insula) 
- Insula models (insula enabled, no piezo)

Features:
- 3-way performance comparison with statistical analysis
- Separate best library loading for piezo and insula models
- ANOVA vs Kruskal-Wallis based on assumption testing
- All pairwise comparisons with multiple comparison corrections
- Enhanced visualizations for 3-group comparisons
- Comprehensive reporting with effect sizes
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import json
import random
from collections import defaultdict
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Add project imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

from run import Runner
import tools
import task
import default
from .interval_production_full_tester import ThreeWayStatisticalAnalyzer


class EnhancedIntervalComparisonFullTester:
    """Unified tester for all three types of interval comparison models"""

    def __init__(self, base_results_dir="interval_comparison_full_results"):
        self.base_results_dir = base_results_dir
        self.dataset_path = "enhanced_interval_datasets/interval_comparison_dataset.json"
        
        # Load best cardiac libraries for both piezo and insula models
        self.best_piezo_library = self._load_best_piezo_library()
        self.best_insula_library = self._load_best_insula_library()
        
        tools.mkdir_p(base_results_dir)

        # Load the dataset
        self.test_data = self._load_test_dataset()

        print(f"Enhanced Interval Comparison Full Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Test dataset: {len(self.test_data)} conditions available")
        print(f"Best piezo library: {self.best_piezo_library}")
        print(f"Best insula library: {self.best_insula_library}")

    def _load_best_piezo_library(self):
        """Load the best performing cardiac library for piezo models"""
        library_results_path = "interval_comparison_cardiac_library_results/cross_model_library_analysis.json"
        
        try:
            with open(library_results_path, 'r') as f:
                library_analysis = json.load(f)
            
            # Get the best library by mean performance
            library_rankings = library_analysis['library_rankings']['mean']
            best_library = max(library_rankings.items(), key=lambda x: x[1])[0]
            
            print(f"Loaded best piezo library from analysis: {best_library}")
            print(f"Best piezo library performance: {library_rankings[best_library]:.4f}")
            
            return best_library
            
        except Exception as e:
            print(f"Warning: Could not load piezo library analysis from {library_results_path}: {e}")
            print("Using default piezo library: hr100_hrv3cal")
            return "hr100_hrv3cal"

    def _load_best_insula_library(self):
        """Load the best performing cardiac library for insula models"""
        library_results_path = "interval_comparison_insula_library_results/cross_insula_model_library_analysis.json"
        
        try:
            with open(library_results_path, 'r') as f:
                library_analysis = json.load(f)
            
            # Get the best library by mean performance
            library_rankings = library_analysis['library_rankings']['mean']
            best_library = max(library_rankings.items(), key=lambda x: x[1])[0]
            
            print(f"Loaded best insula library from analysis: {best_library}")
            print(f"Best insula library performance: {library_rankings[best_library]:.4f}")
            
            return best_library
            
        except Exception as e:
            print(f"Warning: Could not load insula library analysis from {library_results_path}: {e}")
            print("Using default insula library: hr100_hrv3cal")
            return "hr100_hrv3cal"

    def _load_test_dataset(self):
        """Load test conditions from the interval comparison dataset."""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found: {self.dataset_path}")
            print("Creating fallback test conditions...")
            fallback_conditions = []
            for i1 in [1200, 1600, 2000, 2400]:
                for i2 in [1200, 1600, 2000, 2400]:
                    if i1 != i2:  # Only different intervals for comparison
                        for dly in [800, 1200]:
                            fallback_conditions.append({
                                'prod_interval1': i1,
                                'prod_interval2': i2,
                                'dly_interval': dly
                            })
            return fallback_conditions

        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)

        test_data = dataset.get('test', [])
        if not test_data:
            print("No test data found in dataset, using train data sample")
            train_data = dataset.get('train', [])
            if train_data:
                random.seed(42)
                test_data = random.sample(train_data, min(50, len(train_data)))
            else:
                raise ValueError("No usable data found in dataset")

        print(f"Loaded {len(test_data)} test conditions from dataset")
        return test_data

    def select_test_conditions(self, num_conditions=10, seed=42, include_ood=True):
        """Randomly select test conditions from the dataset + out-of-distribution conditions."""
        random.seed(seed)

        # Start with dataset conditions
        if len(self.test_data) <= num_conditions:
            selected = self.test_data.copy()
        else:
            selected = random.sample(self.test_data, num_conditions)

        # Add out-of-distribution conditions if requested
        if include_ood:
            ood_conditions = self._create_ood_conditions()
            selected.extend(ood_conditions)
            print(f"Added {len(ood_conditions)} out-of-distribution test conditions")

        print(f"Selected {len(selected)} total test conditions:")
        print(f"  Dataset conditions: {len(selected) - (len(ood_conditions) if include_ood else 0)}")
        if include_ood:
            print(f"  Out-of-distribution: {len(ood_conditions)}")

        # Show sample conditions
        for i, condition in enumerate(selected[:5]):
            ood_marker = " (OOD)" if include_ood and condition.get('is_ood', False) else ""
            print(
                f"  {i + 1}: I1={condition['prod_interval1']:.0f}ms, I2={condition['prod_interval2']:.0f}ms, dly={condition['dly_interval']:.0f}ms{ood_marker}")
        if len(selected) > 5:
            print(f"  ... and {len(selected) - 5} more")

        return selected

    def _create_ood_conditions(self):
        """Create out-of-distribution test conditions for generalization testing."""
        # Training range was 1200-2400ms, so test beyond this range with BOTH intervals OOD
        ood_intervals_below = [800, 1000, 1100]  # Below training range
        ood_intervals_above = [2500, 2600, 2800, 3000]  # Above training range
        ood_conditions = []

        print(f"Creating out-of-distribution conditions...")
        print(f"  Training range was 1200-2400ms")
        print(f"  Testing extrapolation below: {ood_intervals_below} ms")
        print(f"  Testing extrapolation above: {ood_intervals_above} ms")

        # Both intervals below training range
        for i1 in ood_intervals_below:
            for i2 in ood_intervals_below:
                if i1 != i2:  # Need different intervals for comparison
                    condition = {
                        'prod_interval1': float(i1),
                        'prod_interval2': float(i2),
                        'dly_interval': 1000.0,
                        'is_ood': True,
                        'ood_type': 'both_below_range'
                    }
                    ood_conditions.append(condition)

        # Both intervals above training range
        for i1 in ood_intervals_above:
            for i2 in ood_intervals_above:
                if i1 != i2:  # Need different intervals for comparison
                    condition = {
                        'prod_interval1': float(i1),
                        'prod_interval2': float(i2),
                        'dly_interval': 1000.0,
                        'is_ood': True,
                        'ood_type': 'both_above_range'
                    }
                    ood_conditions.append(condition)

        # Limit to reasonable number of conditions
        return ood_conditions[:12]

    def _determine_model_type(self, analysis):
        """Determine model type based on use_piezo and use_insula flags"""
        use_piezo = analysis.get('use_piezo', False)
        use_insula = analysis.get('use_insula', False)
        
        if use_insula:
            return 'insula'
        elif use_piezo:
            return 'piezo'
        else:
            return 'standard'

    def test_single_model(self, model_dir, rule_name, test_conditions):
        """Test a single model on the selected conditions with retry logic for NaN responses"""
        analysis = tools.analyze_model_directory(model_dir)
        if not analysis['exists']:
            print(f"Model directory not found: {model_dir}")
            return None

        model_name = os.path.basename(model_dir)
        model_type = self._determine_model_type(analysis)
        use_piezo = analysis.get('use_piezo', False)
        use_insula = analysis.get('use_insula', False)

        print(f"\nTesting model: {model_name}")
        print(f"Rule: {rule_name}")
        print(f"Model type: {model_type}")
        print(f"Piezo enabled: {use_piezo}")
        print(f"Insula enabled: {use_insula}")
        print(f"Has pretraining: {analysis.get('pretraining_checkpoint', False)}")

        # Initialize runner
        runner = Runner(
            rule_name=rule_name,
            model_dir=model_dir,
            is_cuda=True,
            noise_on=False
        )

        # Get model info
        model_info = runner.get_model_info()
        print(f"Model info: {model_info}")

        results = {
            'model_dir': model_dir,
            'model_name': model_name,
            'rule_name': rule_name,
            'model_type': model_type,
            'use_piezo': use_piezo,
            'use_insula': use_insula,
            'model_info': model_info,
            'test_results': [],
            'performance_summary': {},
            'nan_counts_per_condition': [],
            'nan_conditions': []  # Track which conditions gave NaN
        }

        # Test each condition with retry logic
        for i, condition in enumerate(test_conditions):
            print(f"  Testing condition {i + 1}/{len(test_conditions)}: I1={condition['prod_interval1']:.0f}ms, I2={condition['prod_interval2']:.0f}ms")

            successful = False
            attempts = 0
            max_attempts = 5

            while not successful and attempts < max_attempts:
                attempts += 1
                try:
                    # Generate cardiac data if needed (piezo or insula model)
                    cardiac_data = None
                    if use_piezo or use_insula:
                        # Calculate task duration for this condition
                        task_duration_ms = condition['prod_interval1'] + condition['prod_interval2'] + condition['dly_interval'] + 2000

                        from dataset import TaskDataset

                        try:
                            import default
                            # Start with a full set of defaults
                            hp = default.get_default_hp(rule_name=rule_name,
                                                        use_piezo=model_info.get("use_piezo", False),
                                                        use_insula=model_info.get("use_insula", False))
                            # Update with model-specific info from saved model
                            hp.update(model_info)

                            # Choose appropriate cardiac data source
                            if use_insula:
                                hp['cardiac_data_source'] = self.best_insula_library
                            elif use_piezo:
                                hp['cardiac_data_source'] = self.best_piezo_library

                            # Build dataset sample with real cardiac data
                            ds = TaskDataset(rule_name, hp, mode="test")
                            sample = ds[0]
                            condition['hb_sequence'] = sample['hb_sequence']
                            if attempts == 1:
                                library_used = self.best_insula_library if use_insula else self.best_piezo_library
                                print(f"    Loaded real cardiac data from dataset (library: {library_used})")
                        except Exception as e:
                            if attempts == 1:
                                print(f"    Could not load cardiac data from dataset: {e}")

                    trial, train_stepper = runner.run(**condition)

                    # Extract comprehensive results and check for NaN
                    outputs = train_stepper.outputs.detach().cpu().numpy()
                    states = [state.detach().cpu().numpy() for state in train_stepper.state_collector]
                    cost = train_stepper.cost.item()

                    # Check for NaN values
                    has_nan_outputs = np.isnan(outputs).any()
                    has_nan_states = any(np.isnan(state).any() for state in states)
                    has_nan_cost = np.isnan(cost)

                    # Calculate performance
                    performance = self._calculate_performance(rule_name, trial, outputs)

                    # Check performance metrics for NaN
                    performance_has_nan = False
                    if 'accuracy' in performance:
                        if np.isnan(performance['accuracy']):
                            performance_has_nan = True

                    has_nan = has_nan_outputs or has_nan_states or has_nan_cost or performance_has_nan

                    if has_nan:
                        print(f"    WARNING: NaN detected in condition {i + 1}, attempt {attempts}")
                        if attempts < max_attempts:
                            print(f"    Retrying... ({max_attempts - attempts} attempts remaining)")
                            continue
                        else:
                            print(f"    Failed after {max_attempts} attempts")
                            results['nan_conditions'].append(
                                f"Condition {i + 1}: I1={condition['prod_interval1']:.0f}ms, I2={condition['prod_interval2']:.0f}ms")
                    else:
                        successful = True
                        if attempts > 1:
                            print(f"    SUCCESS on attempt {attempts}")

                    test_result = {
                        'condition_idx': i,
                        'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                        'outputs': outputs,
                        'states': states,
                        'cost': cost,
                        'trial': {
                            'x': trial.x,
                            'y': trial.y,
                            'cost_mask': trial.cost_mask,
                            'epochs': trial.epochs,
                            'seq_len': trial.seq_len
                        },
                        'cardiac_data': cardiac_data,
                        'performance': performance,
                        'has_nan': has_nan,
                        'nan_details': {
                            'outputs': has_nan_outputs,
                            'states': has_nan_states,
                            'cost': has_nan_cost,
                            'performance': performance_has_nan
                        },
                        'attempts_needed': attempts
                    }

                    results['test_results'].append(test_result)
                    results['nan_counts_per_condition'].append(1 if has_nan else 0)
                    break

                except Exception as e:
                    print(f"    Error testing condition {condition} on attempt {attempts}: {e}")
                    if attempts >= max_attempts:
                        results['nan_counts_per_condition'].append(1)
                        results['nan_conditions'].append(
                            f"Condition {i + 1}: I1={condition['prod_interval1']:.0f}ms, I2={condition['prod_interval2']:.0f}ms (Exception)")
                        break

        # Calculate performance summary
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance(results['test_results'])
            # Add NaN summary
            results['performance_summary']['total_nan_conditions'] = sum(results['nan_counts_per_condition'])
            results['performance_summary']['total_conditions'] = len(test_conditions)
            results['performance_summary']['nan_rate'] = sum(results['nan_counts_per_condition']) / len(test_conditions)

        return results

    def _calculate_performance(self, rule_name, trial, outputs):
        """Calculate performance metrics for interval comparison"""
        if rule_name != 'interval_comparison':
            return {}

        batch_size = outputs.shape[1]

        # Find response period (typically after both intervals)
        if 'go' in trial.epochs:
            response_start = trial.epochs['go'][0]
            response_start = response_start[0] if hasattr(response_start, '__iter__') else response_start
        else:
            # Fallback to after second stimulus
            response_start = trial.epochs.get('stim2', (None, None))[1]
            if response_start is None:
                response_start = int(outputs.shape[0] * 0.8)  # Assume last 20% is response
            else:
                response_start = response_start[0] if hasattr(response_start, '__iter__') else response_start

        # Calculate accuracy
        correct_predictions = 0
        total_predictions = 0

        for b in range(batch_size):
            # Get model outputs during response period
            response_outputs = outputs[response_start:, b, :]  # [time, 2]

            # Get target outputs during response period
            response_targets = trial.y[response_start:, b, :]  # [time, 2]

            if response_outputs.shape[0] == 0 or response_targets.shape[0] == 0:
                continue

            # Average over response period
            mean_output = np.mean(response_outputs, axis=0)  # [2]
            mean_target = np.mean(response_targets, axis=0)  # [2]

            # Model prediction: argmax of outputs
            predicted_class = np.argmax(mean_output)

            # Target class: argmax of targets
            target_class = np.argmax(mean_target)

            if predicted_class == target_class:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        performance = {
            'accuracy': float(accuracy),
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'response_start_time': response_start
        }

        return performance

    def _summarize_performance(self, test_results):
        """Summarize performance across all test conditions"""
        if not test_results:
            return {}

        summary = {
            'num_conditions': len(test_results),
            'mean_cost': np.mean([r['cost'] for r in test_results])
        }

        # Interval comparison task - focus on accuracy
        accuracies = [r['performance']['accuracy'] for r in test_results if
                      'accuracy' in r['performance'] and not np.isnan(r['performance']['accuracy'])]

        if accuracies:
            summary.update({
                'accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'task_type': 'interval_comparison'
            })

        return summary

    def plot_three_way_performance_comparison(self, all_results):
        """Create 3-way performance comparison plot between standard, piezo, and insula models"""
        print("Creating 3-way performance comparison plot...")

        # Separate models by type
        standard_models = [r for r in all_results if r['model_type'] == 'standard']
        piezo_models = [r for r in all_results if r['model_type'] == 'piezo']
        insula_models = [r for r in all_results if r['model_type'] == 'insula']

        if not standard_models or not piezo_models or not insula_models:
            print("Need all three model types for 3-way comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('3-Way Performance Comparison: Standard vs Piezo vs Insula Models (Interval Comparison)', fontsize=16)

        # Calculate mean accuracies for different categories
        def get_accuracy_stats(models, category_filter=None):
            """Get accuracy statistics for models with optional category filter"""
            accuracies = []
            for model in models:
                for test_result in model['test_results']:
                    if test_result['has_nan']:
                        continue
                    if 'accuracy' not in test_result['performance']:
                        continue
                    accuracy = test_result['performance']['accuracy']
                    if np.isnan(accuracy):
                        continue

                    # Apply category filter
                    if category_filter is None:
                        accuracies.append(accuracy)
                    elif category_filter == 'id' and not test_result['condition'].get('is_ood', False):
                        accuracies.append(accuracy)
                    elif category_filter == 'ood' and test_result['condition'].get('is_ood', False):
                        accuracies.append(accuracy)

            return accuracies

        # Plot 1: Overall mean performance
        ax1 = axes[0, 0]
        standard_all = get_accuracy_stats(standard_models)
        piezo_all = get_accuracy_stats(piezo_models)
        insula_all = get_accuracy_stats(insula_models)

        categories = []
        means = []
        stds = []
        colors = []

        if standard_all:
            categories.append('Standard Models')
            means.append(np.mean(standard_all))
            stds.append(np.std(standard_all))
            colors.append('blue')

        if piezo_all:
            categories.append('Piezo Models')
            means.append(np.mean(piezo_all))
            stds.append(np.std(piezo_all))
            colors.append('red')

        if insula_all:
            categories.append('Insula Models')
            means.append(np.mean(insula_all))
            stds.append(np.std(insula_all))
            colors.append('green')

        bars = ax1.bar(categories, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Accuracy')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')
        ax1.set_ylabel('Mean Accuracy')
        ax1.set_title('Overall Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: In-Distribution performance
        ax2 = axes[0, 1]
        standard_id = get_accuracy_stats(standard_models, 'id')
        piezo_id = get_accuracy_stats(piezo_models, 'id')
        insula_id = get_accuracy_stats(insula_models, 'id')

        id_categories = []
        id_means = []
        id_stds = []
        id_colors = []

        if standard_id:
            id_categories.append('Standard ID')
            id_means.append(np.mean(standard_id))
            id_stds.append(np.std(standard_id))
            id_colors.append('blue')

        if piezo_id:
            id_categories.append('Piezo ID')
            id_means.append(np.mean(piezo_id))
            id_stds.append(np.std(piezo_id))
            id_colors.append('red')

        if insula_id:
            id_categories.append('Insula ID')
            id_means.append(np.mean(insula_id))
            id_stds.append(np.std(insula_id))
            id_colors.append('green')

        bars = ax2.bar(id_categories, id_means, yerr=id_stds, color=id_colors, alpha=0.7, capsize=5)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Accuracy')
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')
        ax2.set_ylabel('Mean Accuracy')
        ax2.set_title('In-Distribution Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)

        # Add value labels
        for bar, mean, std in zip(bars, id_means, id_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Out-of-Distribution performance
        ax3 = axes[1, 0]
        standard_ood = get_accuracy_stats(standard_models, 'ood')
        piezo_ood = get_accuracy_stats(piezo_models, 'ood')
        insula_ood = get_accuracy_stats(insula_models, 'ood')

        ood_categories = []
        ood_means = []
        ood_stds = []
        ood_colors = []

        if standard_ood:
            ood_categories.append('Standard OOD')
            ood_means.append(np.mean(standard_ood))
            ood_stds.append(np.std(standard_ood))
            ood_colors.append('blue')

        if piezo_ood:
            ood_categories.append('Piezo OOD')
            ood_means.append(np.mean(piezo_ood))
            ood_stds.append(np.std(piezo_ood))
            ood_colors.append('red')

        if insula_ood:
            ood_categories.append('Insula OOD')
            ood_means.append(np.mean(insula_ood))
            ood_stds.append(np.std(insula_ood))
            ood_colors.append('green')

        bars = ax3.bar(ood_categories, ood_means, yerr=ood_stds, color=ood_colors, alpha=0.7, capsize=5)
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Accuracy')
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')
        ax3.set_ylabel('Mean Accuracy')
        ax3.set_title('Out-of-Distribution Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)

        # Add value labels
        for bar, mean, std in zip(bars, ood_means, ood_stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Sample sizes and failure rates
        ax4 = axes[1, 1]

        # Count successful vs failed responses for OOD
        def count_ood_responses(models, model_type_name):
            total_ood = 0
            failed_ood = 0
            for model in models:
                for test_result in model['test_results']:
                    if test_result['condition'].get('is_ood', False):
                        total_ood += 1
                        if (test_result['has_nan'] or
                                'accuracy' not in test_result['performance'] or
                                np.isnan(test_result['performance']['accuracy'])):
                            failed_ood += 1
            return total_ood, failed_ood

        standard_total_ood, standard_failed_ood = count_ood_responses(standard_models, 'Standard')
        piezo_total_ood, piezo_failed_ood = count_ood_responses(piezo_models, 'Piezo')
        insula_total_ood, insula_failed_ood = count_ood_responses(insula_models, 'Insula')

        # Create grouped bar chart
        labels = ['Standard Models', 'Piezo Models', 'Insula Models']
        successful = [standard_total_ood - standard_failed_ood, 
                     piezo_total_ood - piezo_failed_ood,
                     insula_total_ood - insula_failed_ood]
        failed = [standard_failed_ood, piezo_failed_ood, insula_failed_ood]

        x = np.arange(len(labels))
        width = 0.35

        ax4.bar(x - width / 2, successful, width, label='Successful OOD', color='green', alpha=0.7)
        ax4.bar(x + width / 2, failed, width, label='Failed OOD', color='red', alpha=0.7)

        ax4.set_xlabel('Model Type')
        ax4.set_ylabel('Number of OOD Tests')
        ax4.set_title('OOD Response Success/Failure Rates')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (succ, fail) in enumerate(zip(successful, failed)):
            ax4.text(i - width / 2, succ + 0.5, str(succ), ha='center', va='bottom')
            ax4.text(i + width / 2, fail + 0.5, str(fail), ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, "three_way_performance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3-way performance comparison plot: {plot_path}")

        # Store statistics for report
        self.comparison_stats = {
            'overall': {
                'standard': {'mean': np.mean(standard_all), 'std': np.std(standard_all), 'n': len(standard_all)},
                'piezo': {'mean': np.mean(piezo_all), 'std': np.std(piezo_all), 'n': len(piezo_all)},
                'insula': {'mean': np.mean(insula_all), 'std': np.std(insula_all), 'n': len(insula_all)}
            },
            'in_distribution': {
                'standard': {'mean': np.mean(standard_id) if standard_id else 0, 'std': np.std(standard_id) if standard_id else 0,
                          'n': len(standard_id)},
                'piezo': {'mean': np.mean(piezo_id) if piezo_id else 0, 'std': np.std(piezo_id) if piezo_id else 0,
                         'n': len(piezo_id)},
                'insula': {'mean': np.mean(insula_id) if insula_id else 0, 'std': np.std(insula_id) if insula_id else 0,
                          'n': len(insula_id)}
            },
            'out_of_distribution': {
                'standard': {'mean': np.mean(standard_ood) if standard_ood else 0, 'std': np.std(standard_ood) if standard_ood else 0,
                          'n': len(standard_ood)},
                'piezo': {'mean': np.mean(piezo_ood) if piezo_ood else 0, 'std': np.std(piezo_ood) if piezo_ood else 0,
                         'n': len(piezo_ood)},
                'insula': {'mean': np.mean(insula_ood) if insula_ood else 0, 'std': np.std(insula_ood) if insula_ood else 0,
                          'n': len(insula_ood)}
            },
            'ood_failures': {
                'standard': {'total': standard_total_ood, 'failed': standard_failed_ood},
                'piezo': {'total': piezo_total_ood, 'failed': piezo_failed_ood},
                'insula': {'total': insula_total_ood, 'failed': insula_failed_ood}
            }
        }

    def plot_three_way_scatter(self, all_results):
        """Create scatter plot comparing mean performance across all three model types"""
        print("Creating 3-way model scatter plot...")

        if not all_results:
            print("No results to plot")
            return

        # Separate models by type
        standard_models = [r for r in all_results if r['model_type'] == 'standard']
        piezo_models = [r for r in all_results if r['model_type'] == 'piezo']
        insula_models = [r for r in all_results if r['model_type'] == 'insula']

        if not standard_models or not piezo_models or not insula_models:
            print("Need all three model types for 3-way scatter plot")
            return

        # Extract mean accuracy for each model
        standard_accuracies = []
        piezo_accuracies = []
        insula_accuracies = []

        for model in standard_models:
            if 'accuracy' in model['performance_summary']:
                standard_accuracies.append(model['performance_summary']['accuracy'])

        for model in piezo_models:
            if 'accuracy' in model['performance_summary']:
                piezo_accuracies.append(model['performance_summary']['accuracy'])

        for model in insula_models:
            if 'accuracy' in model['performance_summary']:
                insula_accuracies.append(model['performance_summary']['accuracy'])

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot individual models
        if standard_accuracies:
            ax.scatter([1] * len(standard_accuracies), standard_accuracies,
                       color='blue', s=100, alpha=0.7, label=f'Standard Models (n={len(standard_accuracies)})')

        if piezo_accuracies:
            ax.scatter([2] * len(piezo_accuracies), piezo_accuracies,
                       color='red', s=100, alpha=0.7, label=f'Piezo Models (n={len(piezo_accuracies)})')

        if insula_accuracies:
            ax.scatter([3] * len(insula_accuracies), insula_accuracies,
                       color='green', s=100, alpha=0.7, label=f'Insula Models (n={len(insula_accuracies)})')

        # Add mean lines
        if standard_accuracies:
            standard_mean = np.mean(standard_accuracies)
            ax.axhline(y=standard_mean, color='blue', linestyle='--', alpha=0.8,
                       label=f'Standard Mean: {standard_mean:.3f}')

        if piezo_accuracies:
            piezo_mean = np.mean(piezo_accuracies)
            ax.axhline(y=piezo_mean, color='red', linestyle='--', alpha=0.8,
                       label=f'Piezo Mean: {piezo_mean:.3f}')

        if insula_accuracies:
            insula_mean = np.mean(insula_accuracies)
            ax.axhline(y=insula_mean, color='green', linestyle='--', alpha=0.8,
                       label=f'Insula Mean: {insula_mean:.3f}')

        # Reference lines
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Perfect Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance Level')

        # Formatting
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Standard Models', 'Piezo Models', 'Insula Models'])
        ax.set_ylabel('Mean Accuracy', fontsize=12)
        ax.set_title('3-Way Model Performance Comparison\n(Interval Comparison Task)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.base_results_dir, "three_way_model_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3-way model scatter plot: {save_path}")
        return save_path

    def plot_interval_difficulty_vs_accuracy_three_way(self, all_results, save_path=None):
        """Create scatter plot of interval difficulty vs accuracy for all three model types"""

        # Filter for comparison task results only
        comparison_results = [r for r in all_results if r['rule_name'] == 'interval_comparison']

        if not comparison_results:
            print("No interval comparison results found for scatter plot")
            return

        # Collect data for scatter plot
        standard_data = {}  # difficulty_level: [accuracies]
        piezo_data = {}  # difficulty_level: [accuracies]
        insula_data = {}  # difficulty_level: [accuracies]

        for result in comparison_results:
            model_type = result['model_type']

            for test_result in result['test_results']:
                # Skip if has NaN or missing accuracy data
                if (test_result['has_nan'] or
                        'accuracy' not in test_result['performance'] or
                        np.isnan(test_result['performance']['accuracy'])):
                    continue

                # Calculate difficulty as ratio difference between intervals
                i1 = test_result['condition']['prod_interval1']
                i2 = test_result['condition']['prod_interval2']
                difficulty = abs(i1 - i2) / max(i1, i2)  # Normalized difficulty
                accuracy = test_result['performance']['accuracy']

                # Group by model type
                if model_type == 'standard':
                    if difficulty not in standard_data:
                        standard_data[difficulty] = []
                    standard_data[difficulty].append(accuracy)
                elif model_type == 'piezo':
                    if difficulty not in piezo_data:
                        piezo_data[difficulty] = []
                    piezo_data[difficulty].append(accuracy)
                else:  # insula
                    if difficulty not in insula_data:
                        insula_data[difficulty] = []
                    insula_data[difficulty].append(accuracy)

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot standard models
        if standard_data:
            standard_difficulties = []
            standard_means = []
            standard_stds = []

            for difficulty in sorted(standard_data.keys()):
                accuracies = standard_data[difficulty]
                standard_difficulties.append(difficulty)
                standard_means.append(np.mean(accuracies))
                standard_stds.append(np.std(accuracies))

            # Plot with error bars
            ax.errorbar(standard_difficulties, standard_means, yerr=standard_stds,
                        fmt='o', color='blue', markersize=8, capsize=5,
                        label=f'Standard Models (n={len(standard_data)} difficulty levels)',
                        alpha=0.8, linewidth=2)

        # Plot piezo models
        if piezo_data:
            piezo_difficulties = []
            piezo_means = []
            piezo_stds = []

            for difficulty in sorted(piezo_data.keys()):
                accuracies = piezo_data[difficulty]
                piezo_difficulties.append(difficulty)
                piezo_means.append(np.mean(accuracies))
                piezo_stds.append(np.std(accuracies))

            # Plot with error bars
            ax.errorbar(piezo_difficulties, piezo_means, yerr=piezo_stds,
                        fmt='s', color='red', markersize=8, capsize=5,
                        label=f'Piezo Models (n={len(piezo_data)} difficulty levels)',
                        alpha=0.8, linewidth=2)

        # Plot insula models
        if insula_data:
            insula_difficulties = []
            insula_means = []
            insula_stds = []

            for difficulty in sorted(insula_data.keys()):
                accuracies = insula_data[difficulty]
                insula_difficulties.append(difficulty)
                insula_means.append(np.mean(accuracies))
                insula_stds.append(np.std(accuracies))

            # Plot with error bars
            ax.errorbar(insula_difficulties, insula_means, yerr=insula_stds,
                        fmt='^', color='green', markersize=8, capsize=5,
                        label=f'Insula Models (n={len(insula_data)} difficulty levels)',
                        alpha=0.8, linewidth=2)

        # Add reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 1.02, 'Perfect Accuracy',
                horizontalalignment='right', fontsize=10, color='gray')
        ax.text(ax.get_xlim()[1] * 0.95, 0.52, 'Chance Level',
                horizontalalignment='right', fontsize=10, color='gray')

        # Formatting
        ax.set_xlabel('Task Difficulty (Normalized Interval Difference)', fontsize=12)
        ax.set_ylabel('Mean Accuracy', fontsize=12)
        ax.set_title('Interval Difficulty vs Accuracy - 3-Way Comparison\n(Interval Comparison Task)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)

        # Add summary statistics as text
        summary_text = ""
        if standard_data and piezo_data and insula_data:
            standard_overall_mean = np.mean([np.mean(accuracies) for accuracies in standard_data.values()])
            piezo_overall_mean = np.mean([np.mean(accuracies) for accuracies in piezo_data.values()])
            insula_overall_mean = np.mean([np.mean(accuracies) for accuracies in insula_data.values()])

            summary_text = f"Overall Mean Accuracy:\n"
            summary_text += f"  Standard: {standard_overall_mean:.3f}\n"
            summary_text += f"  Piezo: {piezo_overall_mean:.3f}\n"
            summary_text += f"  Insula: {insula_overall_mean:.3f}"

            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(self.base_results_dir, "interval_difficulty_vs_accuracy_three_way_scatter.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3-way interval difficulty vs accuracy scatter plot: {save_path}")

        # Print summary statistics
        print(f"\n3-Way Scatter Plot Summary:")
        print(f"  Standard models: {len(standard_data)} different difficulty levels")
        print(f"  Piezo models: {len(piezo_data)} different difficulty levels")
        print(f"  Insula models: {len(insula_data)} different difficulty levels")

        if standard_data:
            total_standard_points = sum(len(accuracies) for accuracies in standard_data.values())
            print(f"  Total standard data points: {total_standard_points}")

        if piezo_data:
            total_piezo_points = sum(len(accuracies) for accuracies in piezo_data.values())
            print(f"  Total piezo data points: {total_piezo_points}")

        if insula_data:
            total_insula_points = sum(len(accuracies) for accuracies in insula_data.values())
            print(f"  Total insula data points: {total_insula_points}")

    def discover_all_comparison_models(self):
        """Discover all trained interval comparison models (standard, piezo, and insula)"""
        model_dirs = []

        # Define possible base directories for all model types
        possible_base_dirs = [
            'enhanced_piezo_comparison_results',  # All three model types
            'enhanced_insula_comparison_results',  # Insula models  
            'comparison_results',  # Mixed
            'interval_comparison_results',  # Mixed
            'model/interval_comparison',  # Mixed
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                print(f"Searching in {base_dir}...")
                self._search_model_subdirs(base_dir, 'interval_comparison', model_dirs)

        return model_dirs

    def _search_model_subdirs(self, base_dir, rule_name, model_dirs):
        """Search for model subdirectories, categorizing by type"""
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Check if this looks like a run directory
                    if ('run_' in item or 'insula_run_' in item or 'piezo_run_' in item or 
                        'no_piezo_run_' in item or 'no_insula' in item or 'baseline_run_' in item or
                        item.startswith('w2_') or item.isdigit()):
                        # Find the best checkpoint in this run directory
                        best_model_path = self._find_best_model_checkpoint(item_path)
                        if best_model_path:
                            model_dirs.append((best_model_path, rule_name, item))
                            print(f"  Found model: {item}")
                    elif self._has_valid_model(item_path):
                        # This directory itself contains a model
                        model_dirs.append((item_path, rule_name, item))
                        print(f"  Found model: {item}")
        except PermissionError:
            print(f"Permission denied accessing {base_dir}")
        except Exception as e:
            print(f"Error searching {base_dir}: {e}")

    def _find_best_model_checkpoint(self, base_model_dir, use_best_accuracy=True):
        """Find the best model checkpoint in a run directory based on accuracy"""
        # Priority order: finalResult > best accuracy checkpoint > highest numbered checkpoint > main directory
        final_path = os.path.join(base_model_dir, 'finalResult')
        if os.path.exists(final_path) and self._has_valid_model(final_path):
            return final_path

        # Check for numbered checkpoints
        numbered_checkpoints = []
        if os.path.exists(base_model_dir):
            for item in os.listdir(base_model_dir):
                if item.isdigit():
                    checkpoint_path = os.path.join(base_model_dir, item)
                    if os.path.isdir(checkpoint_path) and self._has_valid_model(checkpoint_path):
                        numbered_checkpoints.append((int(item), checkpoint_path))

        if numbered_checkpoints and use_best_accuracy:
            # Try to find the best checkpoint based on accuracy from log.json
            best_checkpoint = self._find_best_accuracy_checkpoint(base_model_dir, numbered_checkpoints)
            if best_checkpoint:
                return best_checkpoint

        # Fallback: Get highest numbered checkpoint
        if numbered_checkpoints:
            numbered_checkpoints.sort(reverse=True)  # Highest number first
            return numbered_checkpoints[0][1]

        # Check main directory as fallback
        if self._has_valid_model(base_model_dir):
            return base_model_dir

        return None

    def _find_best_accuracy_checkpoint(self, base_model_dir, numbered_checkpoints):
        """Find checkpoint with highest accuracy based on individual checkpoint log.json files"""
        try:
            max_accuracy = -float('inf')
            best_checkpoint_path = None
            best_checkpoint_num = -1
            
            # Check each numbered checkpoint's individual log.json file
            for checkpoint_num, checkpoint_path in numbered_checkpoints:
                checkpoint_log_path = os.path.join(checkpoint_path, 'log.json')
                
                if not os.path.exists(checkpoint_log_path):
                    continue
                
                try:
                    with open(checkpoint_log_path, 'r') as f:
                        checkpoint_log = json.load(f)
                    
                    # Look for accuracy metric (success_action_prob)
                    if 'success_action_prob' in checkpoint_log:
                        accuracy = checkpoint_log['success_action_prob']
                        
                        if np.isfinite(accuracy) and accuracy > max_accuracy:
                            max_accuracy = accuracy
                            best_checkpoint_path = checkpoint_path
                            best_checkpoint_num = checkpoint_num
                            
                except Exception as e:
                    print(f"    Warning: Could not read {checkpoint_log_path}: {e}")
                    continue
            
            if best_checkpoint_path:
                print(f"  Selected best accuracy checkpoint: {best_checkpoint_num} (accuracy: {max_accuracy:.6f})")
                return best_checkpoint_path
            else:
                print(f"  Warning: No accuracy data found in checkpoint logs")
                return None
            
        except Exception as e:
            print(f"  Warning: Could not determine best checkpoint from accuracy: {e}")
            return None

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)

    def _generate_enhanced_summary_report(self, all_results, test_conditions):
        """Generate enhanced summary report with 3-way model statistics and NaN tracking"""
        report_path = os.path.join(self.base_results_dir, "enhanced_full_comparison_test_report.txt")

        # Separate conditions
        id_conditions = [c for c in test_conditions if not c.get('is_ood', False)]
        ood_conditions = [c for c in test_conditions if c.get('is_ood', False)]

        # Separate models by type
        standard_models = [r for r in all_results if r['model_type'] == 'standard']
        piezo_models = [r for r in all_results if r['model_type'] == 'piezo']
        insula_models = [r for r in all_results if r['model_type'] == 'insula']

        # Collect NaN information
        all_nan_conditions = []
        for result in all_results:
            if 'nan_conditions' in result and result['nan_conditions']:
                all_nan_conditions.append({
                    'model_name': result['model_name'],
                    'model_type': result['model_type'],
                    'nan_conditions': result['nan_conditions']
                })

        with open(report_path, 'w') as f:
            f.write("ENHANCED INTERVAL COMPARISON FULL DATASET TEST REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Conditions: {len(test_conditions)} total\n")
            f.write(f"  In-Distribution: {len(id_conditions)} samples from dataset\n")
            f.write(f"  Out-of-Distribution: {len(ood_conditions)} extrapolation tests\n")
            f.write(f"Models Tested: {len(all_results)} total\n")

            # Model breakdown by type
            f.write(f"Standard Models: {len(standard_models)}\n")
            f.write(f"Piezo Models: {len(piezo_models)}\n")
            f.write(f"Insula Models: {len(insula_models)}\n\n")

            # Library information
            f.write(f"CARDIAC LIBRARY USAGE:\n")
            f.write(f"Best Piezo Library: {self.best_piezo_library}\n")
            f.write(f"Best Insula Library: {self.best_insula_library}\n\n")

            # NaN CONDITIONS REPORT
            f.write("NaN CONDITIONS REPORT\n")
            f.write("-" * 25 + "\n")
            if all_nan_conditions:
                for nan_info in all_nan_conditions:
                    f.write(f"Model: {nan_info['model_name']} ({nan_info['model_type']})\n")
                    for condition in nan_info['nan_conditions']:
                        f.write(f"  - {condition}\n")
                    f.write("\n")
            else:
                f.write("No models produced NaN responses\n\n")

            # Enhanced 3-way performance comparison
            f.write("3-WAY PERFORMANCE COMPARISON (ALL GRAPHED METRICS)\n")
            f.write("-" * 60 + "\n")

            if hasattr(self, 'comparison_stats'):
                stats = self.comparison_stats

                # Overall performance
                f.write(f"OVERALL PERFORMANCE (from 3-Way Performance Comparison):\n")
                f.write(f"  Standard Models: {stats['overall']['standard']['mean']:.4f} ± {stats['overall']['standard']['std']:.4f} (n={stats['overall']['standard']['n']})\n")
                f.write(f"  Piezo Models:    {stats['overall']['piezo']['mean']:.4f} ± {stats['overall']['piezo']['std']:.4f} (n={stats['overall']['piezo']['n']})\n")
                f.write(f"  Insula Models:   {stats['overall']['insula']['mean']:.4f} ± {stats['overall']['insula']['std']:.4f} (n={stats['overall']['insula']['n']})\n")
                f.write("\n")

                # In-Distribution performance
                f.write(f"IN-DISTRIBUTION PERFORMANCE (from 3-Way Performance Comparison):\n")
                f.write(f"  Standard Models: {stats['in_distribution']['standard']['mean']:.4f} ± {stats['in_distribution']['standard']['std']:.4f} (n={stats['in_distribution']['standard']['n']})\n")
                f.write(f"  Piezo Models:    {stats['in_distribution']['piezo']['mean']:.4f} ± {stats['in_distribution']['piezo']['std']:.4f} (n={stats['in_distribution']['piezo']['n']})\n")
                f.write(f"  Insula Models:   {stats['in_distribution']['insula']['mean']:.4f} ± {stats['in_distribution']['insula']['std']:.4f} (n={stats['in_distribution']['insula']['n']})\n")
                f.write("\n")

                # Out-of-Distribution performance
                f.write(f"OUT-OF-DISTRIBUTION PERFORMANCE (from 3-Way Performance Comparison):\n")
                f.write(f"  Standard Models: {stats['out_of_distribution']['standard']['mean']:.4f} ± {stats['out_of_distribution']['standard']['std']:.4f} (n={stats['out_of_distribution']['standard']['n']})\n")
                f.write(f"  Piezo Models:    {stats['out_of_distribution']['piezo']['mean']:.4f} ± {stats['out_of_distribution']['piezo']['std']:.4f} (n={stats['out_of_distribution']['piezo']['n']})\n")
                f.write(f"  Insula Models:   {stats['out_of_distribution']['insula']['mean']:.4f} ± {stats['out_of_distribution']['insula']['std']:.4f} (n={stats['out_of_distribution']['insula']['n']})\n")
                f.write("\n")

                # OOD Failure Analysis
                f.write(f"OUT-OF-DISTRIBUTION FAILURE ANALYSIS (from 3-Way Performance Comparison):\n")
                standard_fail_rate = stats['ood_failures']['standard']['failed'] / stats['ood_failures']['standard']['total'] if stats['ood_failures']['standard']['total'] > 0 else 0
                piezo_fail_rate = stats['ood_failures']['piezo']['failed'] / stats['ood_failures']['piezo']['total'] if stats['ood_failures']['piezo']['total'] > 0 else 0
                insula_fail_rate = stats['ood_failures']['insula']['failed'] / stats['ood_failures']['insula']['total'] if stats['ood_failures']['insula']['total'] > 0 else 0

                f.write(f"  Standard Models: {stats['ood_failures']['standard']['failed']}/{stats['ood_failures']['standard']['total']} failed ({standard_fail_rate:.1%})\n")
                f.write(f"  Piezo Models:    {stats['ood_failures']['piezo']['failed']}/{stats['ood_failures']['piezo']['total']} failed ({piezo_fail_rate:.1%})\n")
                f.write(f"  Insula Models:   {stats['ood_failures']['insula']['failed']}/{stats['ood_failures']['insula']['total']} failed ({insula_fail_rate:.1%})\n")
                f.write("\n")

            # Individual model results by type
            f.write("INDIVIDUAL MODEL RESULTS BY TYPE\n")
            f.write("-" * 35 + "\n")

            for model_type, models in [('Standard', standard_models), ('Piezo', piezo_models), ('Insula', insula_models)]:
                f.write(f"\n{model_type.upper()} MODELS:\n")
                f.write("-" * (len(model_type) + 8) + "\n")

                for result in models:
                    f.write(f"Model: {result['model_name']}\n")
                    f.write(f"Task: {result['rule_name']}\n")
                    f.write(f"Performance Summary:\n")

                    for key, value in result['performance_summary'].items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")

                    # Add NaN info for this model
                    if 'nan_conditions' in result and result['nan_conditions']:
                        f.write(f"NaN Conditions:\n")
                        for condition in result['nan_conditions']:
                            f.write(f"  - {condition}\n")

                    f.write("\n")

        print(f"Generated enhanced 3-way summary report: {report_path}")

    def run_enhanced_full_dataset_test(self, num_conditions=15, include_ood=True):
        """Run the complete enhanced dataset-driven test for all three model types"""
        print("Starting Enhanced Interval Comparison Full Dataset Test")
        print("=" * 80)

        # Discover all models
        model_directories = self.discover_all_comparison_models()
        if not model_directories:
            print("No interval comparison models found!")
            return

        print(f"Found {len(model_directories)} models to test")

        # Select test conditions from dataset + OOD
        test_conditions = self.select_test_conditions(num_conditions, include_ood=include_ood)

        # Test each model
        all_results = []
        for model_info in model_directories:
            if len(model_info) == 3:
                model_dir, rule_name, run_name = model_info
            else:
                model_dir, rule_name = model_info
                run_name = os.path.basename(model_dir)

            print(f"\nTesting {run_name} ({rule_name})")
            results = self.test_single_model(model_dir, rule_name, test_conditions)

            if results:
                results['model_name'] = run_name
                all_results.append(results)

                # Save individual results
                results_path = os.path.join(self.base_results_dir, f"{run_name}_dataset_test.pkl")
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Saved results: {results_path}")

        # Create enhanced visualizations
        if all_results:
            # 3-way performance comparison plot
            self.plot_three_way_performance_comparison(all_results)

            # 3-way model scatter plot
            self.plot_three_way_scatter(all_results)

            # 3-way interval difficulty vs accuracy scatter plot
            self.plot_interval_difficulty_vs_accuracy_three_way(all_results)

            # Generate statistical analysis
            analyzer = ThreeWayStatisticalAnalyzer()
            standard_data, piezo_data, insula_data = analyzer.extract_performance_data(all_results, 'interval_comparison')
            statistical_results = analyzer.perform_three_way_statistical_tests(standard_data, piezo_data, insula_data, 'interval_comparison')
            
            # Generate and save statistical report
            stat_report = analyzer.generate_three_way_statistical_report(statistical_results, 'interval_comparison')
            stat_report_path = os.path.join(self.base_results_dir, "three_way_statistical_analysis.txt")
            with open(stat_report_path, 'w') as f:
                f.write(stat_report)
            print(f"Generated 3-way statistical analysis: {stat_report_path}")

            # Generate enhanced summary report
            self._generate_enhanced_summary_report(all_results, test_conditions)

            # Save consolidated results
            summary_path = os.path.join(self.base_results_dir, "all_enhanced_full_dataset_test_results.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

        print(f"\nEnhanced full dataset test complete!")
        print(f"Results saved to: {self.base_results_dir}")
        print(f"Generated plots:")
        print(f"  - 3-way performance comparison: three_way_performance_comparison.png")
        print(f"  - 3-way model scatter: three_way_model_scatter.png")
        print(f"  - 3-way interval difficulty vs accuracy: interval_difficulty_vs_accuracy_three_way_scatter.png")
        print(f"Generated reports:")
        print(f"  - 3-way statistical analysis: three_way_statistical_analysis.txt")
        print(f"  - Enhanced summary report: enhanced_full_comparison_test_report.txt")

        return all_results


def main():
    """Run enhanced interval comparison full dataset test with 3-way statistical analysis"""
    tester = EnhancedIntervalComparisonFullTester()
    results = tester.run_enhanced_full_dataset_test(num_conditions=25, include_ood=True)

    print(f"\nEnhanced full dataset testing with 3-way statistical analysis complete!")
    print(f"Results saved to: {tester.base_results_dir}")


if __name__ == "__main__":
    main()
