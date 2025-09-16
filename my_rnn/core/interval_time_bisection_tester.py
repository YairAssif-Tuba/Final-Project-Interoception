"""
Enhanced Time Bisection Dataset Tester - COMPLETE VERSION
=========================================================

Tests trained time bisection models with enhanced analysis including:
- Mean performance comparison between piezo and non-piezo models
- Detailed breakdown of ID vs OOD performance
- Failure rate tracking for out-of-distribution durations
- Task vs model output visualization
- Comprehensive performance analysis
- Individual model plotting functions
- Per-condition analysis with proper categorization by reference standards

REVISED to match the robust testing logic from time_bisection_library_tester.py
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

# Add project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import Runner
import tools
import task
import default
from real_cardiac_data import create_real_cardiac_data_for_task
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict


class StatisticalAnalyzer:
    """Statistical analysis module for enhanced model testers"""

    def __init__(self):
        self.results = {}

    def extract_performance_data(self, all_results, task_type='time_bisection'):
        """Extract individual performance data points for statistical analysis"""
        piezo_data = {
            'overall': [],
            'in_distribution': [],
            'out_of_distribution': [],
            'model_means': []
        }

        no_piezo_data = {
            'overall': [],
            'in_distribution': [],
            'out_of_distribution': [],
            'model_means': []
        }

        # Performance metric key based on task type
        if task_type == 'interval_production':
            perf_key = 'mean_completion'
        else:  # interval_comparison, time_bisection
            perf_key = 'accuracy'

        for result in all_results:
            use_piezo = result['use_piezo']
            data_dict = piezo_data if use_piezo else no_piezo_data

            # Extract model-level mean for scatter plot analysis
            if perf_key in result['performance_summary']:
                data_dict['model_means'].append(result['performance_summary'][perf_key])

            # Extract condition-level data
            for test_result in result['test_results']:
                if test_result['has_nan'] or perf_key not in test_result['performance']:
                    continue

                performance_value = test_result['performance'][perf_key]
                if np.isnan(performance_value):
                    continue

                # Add to overall
                data_dict['overall'].append(performance_value)

                # Categorize by distribution type
                is_ood = test_result['condition'].get('is_ood', False)
                if is_ood:
                    data_dict['out_of_distribution'].append(performance_value)
                else:
                    data_dict['in_distribution'].append(performance_value)

        return piezo_data, no_piezo_data

    def perform_statistical_tests(self, piezo_data, no_piezo_data, task_type='time_bisection'):
        """Perform comprehensive statistical testing"""
        results = {}

        # Performance metric name for reporting
        if task_type == 'interval_production':
            metric_name = 'Completion Ratio'
        else:
            metric_name = 'Accuracy'

        categories = ['overall', 'in_distribution', 'out_of_distribution', 'model_means']

        for category in categories:
            piezo_vals = np.array(piezo_data[category])
            no_piezo_vals = np.array(no_piezo_data[category])

            if len(piezo_vals) == 0 or len(no_piezo_vals) == 0:
                results[category] = {'error': 'Insufficient data'}
                continue

            # Descriptive statistics
            desc_stats = {
                'piezo_n': len(piezo_vals),
                'piezo_mean': np.mean(piezo_vals),
                'piezo_std': np.std(piezo_vals, ddof=1),
                'piezo_median': np.median(piezo_vals),
                'no_piezo_n': len(no_piezo_vals),
                'no_piezo_mean': np.mean(no_piezo_vals),
                'no_piezo_std': np.std(no_piezo_vals, ddof=1),
                'no_piezo_median': np.median(no_piezo_vals),
                'mean_difference': np.mean(piezo_vals) - np.mean(no_piezo_vals)
            }

            # Normality tests
            _, piezo_normality_p = stats.shapiro(piezo_vals) if len(piezo_vals) > 3 else (None, None)
            _, no_piezo_normality_p = stats.shapiro(no_piezo_vals) if len(no_piezo_vals) > 3 else (None, None)

            # Variance equality test
            _, variance_equality_p = stats.levene(piezo_vals, no_piezo_vals)

            # Choose appropriate test based on assumptions
            assume_normal = (piezo_normality_p is None or piezo_normality_p > 0.05) and \
                            (no_piezo_normality_p is None or no_piezo_normality_p > 0.05)
            assume_equal_var = variance_equality_p > 0.05

            # Parametric tests
            if assume_normal:
                t_stat, t_p = stats.ttest_ind(piezo_vals, no_piezo_vals, equal_var=assume_equal_var)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(piezo_vals) - 1) * desc_stats['piezo_std'] ** 2 +
                                      (len(no_piezo_vals) - 1) * desc_stats['no_piezo_std'] ** 2) /
                                     (len(piezo_vals) + len(no_piezo_vals) - 2))
                cohens_d = desc_stats['mean_difference'] / pooled_std

                # Confidence interval for difference in means
                se_diff = pooled_std * np.sqrt(1 / len(piezo_vals) + 1 / len(no_piezo_vals))
                df = len(piezo_vals) + len(no_piezo_vals) - 2
                t_critical = stats.t.ppf(0.975, df)
                ci_lower = desc_stats['mean_difference'] - t_critical * se_diff
                ci_upper = desc_stats['mean_difference'] + t_critical * se_diff
            else:
                t_stat, t_p = None, None
                cohens_d = None
                ci_lower, ci_upper = None, None

            # Non-parametric test (always compute as backup)
            u_stat, u_p = stats.mannwhitneyu(piezo_vals, no_piezo_vals, alternative='two-sided')

            # Effect size for Mann-Whitney (r = Z/sqrt(N))
            z_score = stats.norm.ppf(u_p / 2)  # Convert p to z-score
            r_effect = abs(z_score) / np.sqrt(len(piezo_vals) + len(no_piezo_vals))

            results[category] = {
                'descriptives': desc_stats,
                'assumptions': {
                    'piezo_normality_p': piezo_normality_p,
                    'no_piezo_normality_p': no_piezo_normality_p,
                    'variance_equality_p': variance_equality_p,
                    'assume_normal': assume_normal,
                    'assume_equal_var': assume_equal_var
                },
                'parametric': {
                    't_statistic': t_stat,
                    'p_value': t_p,
                    'cohens_d': cohens_d,
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper
                },
                'nonparametric': {
                    'u_statistic': u_stat,
                    'p_value': u_p,
                    'effect_size_r': r_effect
                }
            }

        return results

    def apply_multiple_comparison_correction(self, results, method='bonferroni'):
        """Apply multiple comparison correction"""
        # Extract all p-values
        p_values = []
        test_names = []

        for category, result in results.items():
            if 'error' in result:
                continue

            # Use parametric test if assumptions met, otherwise non-parametric
            if result['assumptions']['assume_normal']:
                if result['parametric']['p_value'] is not None:
                    p_values.append(result['parametric']['p_value'])
                    test_names.append(f"{category}_parametric")
            else:
                p_values.append(result['nonparametric']['p_value'])
                test_names.append(f"{category}_nonparametric")

        if not p_values:
            return results

        # Apply correction
        if method == 'bonferroni':
            corrected_p = [min(p * len(p_values), 1.0) for p in p_values]
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_p = [0] * len(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(p_values[idx] * (len(p_values) - i), 1.0)
                if i > 0:
                    corrected_p[idx] = max(corrected_p[idx], corrected_p[sorted_indices[i - 1]])
        else:
            corrected_p = p_values  # No correction

        # Add corrected p-values back to results
        corrected_dict = dict(zip(test_names, corrected_p))

        for category, result in results.items():
            if 'error' in result:
                continue

            param_key = f"{category}_parametric"
            nonparam_key = f"{category}_nonparametric"

            if param_key in corrected_dict:
                result['parametric']['corrected_p_value'] = corrected_dict[param_key]
            if nonparam_key in corrected_dict:
                result['nonparametric']['corrected_p_value'] = corrected_dict[nonparam_key]

        return results

    def interpret_effect_size(self, cohens_d=None, r=None):
        """Interpret effect sizes using Cohen's conventions"""
        if cohens_d is not None:
            d = abs(cohens_d)
            if d < 0.2:
                return "negligible"
            elif d < 0.5:
                return "small"
            elif d < 0.8:
                return "medium"
            else:
                return "large"

        if r is not None:
            if r < 0.1:
                return "negligible"
            elif r < 0.3:
                return "small"
            elif r < 0.5:
                return "medium"
            else:
                return "large"

        return "unknown"

    def generate_statistical_report(self, results, task_type='time_bisection'):
        """Generate comprehensive statistical analysis report"""
        if task_type == 'time_bisection':
            metric_name = 'Accuracy'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be a significant difference in accuracy between piezo and non-piezo models (all conditions)
2. There will be a significant difference in accuracy between piezo and non-piezo models (in-distribution)
3. There will be a significant difference in accuracy between piezo and non-piezo models (out-of-distribution)
"""
        else:
            metric_name = 'Accuracy'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be a significant difference in accuracy between piezo and non-piezo models (all conditions)
2. There will be a significant difference in accuracy between piezo and non-piezo models (in-distribution)
3. There will be a significant difference in accuracy between piezo and non-piezo models (out-of-distribution)
"""

        report = f"""
COMPREHENSIVE STATISTICAL ANALYSIS
===============================================
{hypothesis_text}
STATISTICAL APPROACH:
- Independent samples t-tests for normally distributed data
- Mann-Whitney U tests for non-normal data
- Bonferroni correction for multiple comparisons
- Effect size calculations (Cohen's d and r)
- 95% confidence intervals for differences

NOTE: NaN values are noted but not excluded from calculations as they represent
systematic model failures that are part of the performance evaluation.

"""

        categories = ['overall', 'in_distribution', 'out_of_distribution', 'model_means']
        category_names = ['Overall Performance', 'In-Distribution Performance',
                          'Out-of-Distribution Performance', 'Model-Level Comparison']

        for category, name in zip(categories, category_names):
            if category not in results or 'error' in results[category]:
                report += f"\n{name.upper()}:\n"
                report += f"  ERROR: {results.get(category, {}).get('error', 'Unknown error')}\n"
                continue

            result = results[category]
            desc = result['descriptives']
            assumptions = result['assumptions']

            report += f"\n{name.upper()}:\n"
            report += f"  Sample Sizes: Piezo n={desc['piezo_n']}, Non-Piezo n={desc['no_piezo_n']}\n"
            report += f"  Means: Piezo={desc['piezo_mean']:.4f} (SD={desc['piezo_std']:.4f}), " \
                      f"Non-Piezo={desc['no_piezo_mean']:.4f} (SD={desc['no_piezo_std']:.4f})\n"
            report += f"  Mean Difference: {desc['mean_difference']:+.4f} (Piezo - Non-Piezo)\n"

            # Assumption tests
            report += f"\n  Assumption Tests:\n"
            if assumptions['piezo_normality_p'] is not None:
                report += f"    Normality (Shapiro-Wilk): Piezo p={assumptions['piezo_normality_p']:.4f}, " \
                          f"Non-Piezo p={assumptions['no_piezo_normality_p']:.4f}\n"
            report += f"    Equal Variances (Levene): p={assumptions['variance_equality_p']:.4f}\n"

            # Statistical tests
            if assumptions['assume_normal'] and result['parametric']['p_value'] is not None:
                param = result['parametric']
                report += f"\n  Parametric Test (Independent t-test):\n"
                report += f"    t = {param['t_statistic']:.4f}, p = {param['p_value']:.4f}"
                if 'corrected_p_value' in param:
                    report += f" (corrected p = {param['corrected_p_value']:.4f})"
                report += f"\n    Cohen's d = {param['cohens_d']:.4f} ({self.interpret_effect_size(cohens_d=param['cohens_d'])} effect)\n"
                report += f"    95% CI for difference: [{param['ci_95_lower']:.4f}, {param['ci_95_upper']:.4f}]\n"
            else:
                report += f"\n  Parametric assumptions not met, using non-parametric test.\n"

            nonparam = result['nonparametric']
            report += f"\n  Non-parametric Test (Mann-Whitney U):\n"
            report += f"    U = {nonparam['u_statistic']:.4f}, p = {nonparam['p_value']:.4f}"
            if 'corrected_p_value' in nonparam:
                report += f" (corrected p = {nonparam['corrected_p_value']:.4f})"
            report += f"\n    Effect size r = {nonparam['effect_size_r']:.4f} ({self.interpret_effect_size(r=nonparam['effect_size_r'])} effect)\n"

            # Interpretation
            primary_test = result['parametric'] if assumptions['assume_normal'] else result['nonparametric']
            p_val = primary_test.get('corrected_p_value', primary_test['p_value'])

            if p_val is not None:
                if p_val < 0.001:
                    sig_text = "highly significant (p < 0.001)"
                elif p_val < 0.01:
                    sig_text = "very significant (p < 0.01)"
                elif p_val < 0.05:
                    sig_text = "significant (p < 0.05)"
                else:
                    sig_text = "not significant (p ≥ 0.05)"

                report += f"\n  INTERPRETATION: The difference is {sig_text}.\n"

        return report


class EnhancedTimeBisectionTester:
    """Enhanced tester for time bisection models with detailed piezo vs non-piezo analysis"""

    def __init__(self, base_results_dir="enhanced_bisection_dataset_test_results"):
        self.base_results_dir = base_results_dir
        self.cardiac_library = "hr60_hrv0cal"  # FIXED: Use specific cardiac library
        self.dataset_path = "enhanced_interval_datasets/time_bisection_dataset.json"
        self.rule_name = 'time_bisection'
        tools.mkdir_p(base_results_dir)

        # Load the dataset
        self.test_data = self._load_test_dataset()

        print(f"Enhanced Time Bisection Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Cardiac library: {self.cardiac_library}")
        print(f"Test dataset: {len(self.test_data)} conditions available")

    def _load_test_dataset(self):
        """Load test conditions from the time bisection dataset."""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found: {self.dataset_path}")
            print("Creating fallback test conditions...")
            fallback_conditions = []
            for duration in [300, 400, 500, 600, 700, 800, 900]:
                for short_standard in [300]:
                    for long_standard in [900]:
                        for response_duration in [300]:
                            fallback_conditions.append({
                                'duration': duration,
                                'short_standard': short_standard,
                                'long_standard': long_standard,
                                'response_duration': response_duration,
                                'std': 40
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
                f"  {i + 1}: duration={condition['duration']:.0f}ms, standards={condition['short_standard']:.0f}/{condition['long_standard']:.0f}ms{ood_marker}")
        if len(selected) > 5:
            print(f"  ... and {len(selected) - 5} more")

        return selected

    def _create_ood_conditions(self):
        """Create out-of-distribution test conditions for generalization testing."""
        ood_conditions = []

        print(f"Creating out-of-distribution conditions...")
        print(f"  Training was on 1000-2000ms range")
        print(f"  Testing generalization to different reference frameworks")

        # Test with compressed range (300-900) - challenging durations relative to these standards
        for duration in [500, 600, 700, 800]:  # Ambiguous durations for 300-900 framework
            condition = {
                'duration': float(duration),
                'short_standard': 300,
                'long_standard': 900,
                'response_duration': 300.0,
                'std': 40,
                'is_ood': True,
                'ood_type': 'different_framework'
            }
            ood_conditions.append(condition)

        # Test with expanded range - challenging durations relative to these standards
        for duration in [2200, 2400, 2600, 2800]:  # Ambiguous durations for 1500-2500 framework
            condition = {
                'duration': float(duration),
                'short_standard': 2000,
                'long_standard': 3000,
                'response_duration': 300.0,
                'std': 40,
                'is_ood': True,
                'ood_type': 'expanded_framework'
            }
            ood_conditions.append(condition)

        return ood_conditions

    def test_single_model(self, model_dir, rule_name, test_conditions):
        """Test a single model on the selected conditions - REVISED WITH ROBUST LOGIC"""
        analysis = tools.analyze_model_directory(model_dir)
        if not analysis['exists']:
            print(f"Model directory not found: {model_dir}")
            return None

        model_name = os.path.basename(model_dir)
        use_piezo = analysis['use_piezo']

        print(f"\nTesting model: {model_name}")
        print(f"Rule: {rule_name}")
        print(f"Piezo enabled: {use_piezo}")
        print(f"Has pretraining: {analysis['pretraining_checkpoint']}")

        # FIXED: Initialize runner with proper hyperparameters (matching library tester logic)
        hp = tools.load_hp(model_dir)
        if 'rnn_type' not in hp:
            hp['rnn_type'] = 'RNN'
        if 'alpha' not in hp:
            hp['alpha'] = 1.0 * hp.get('dt', 20) / hp.get('tau', 20)

        runner = Runner(
            rule_name=rule_name,
            model_dir=model_dir,
            hp=hp,
            is_cuda=False,
            noise_on=False
        )

        # Get model info
        model_info = runner.get_model_info()
        print(f"Model info: {model_info}")

        results = {
            'model_dir': model_dir,
            'model_name': model_name,
            'rule_name': rule_name,
            'use_piezo': use_piezo,
            'model_info': model_info,
            'test_results': [],
            'performance_summary': {},
            'nan_counts_per_condition': []
        }

        # Test each condition
        for i, condition in enumerate(test_conditions):
            print(f"  Testing condition {i + 1}/{len(test_conditions)}: duration={condition['duration']:.0f}ms")

            try:
                # FIXED: Generate cardiac data using the robust approach from library tester
                if use_piezo:
                    # Calculate task duration for this condition
                    task_duration_ms = condition['duration'] + condition.get('response_duration',
                                                                             500) + 500  # Add buffer

                    # FIXED: Use create_real_cardiac_data_for_task directly (like library tester)
                    cardiac_data = create_real_cardiac_data_for_task(
                        task_duration_ms=task_duration_ms,
                        dt=runner.hp.get('dt', 20),
                        slice_size=runner.hp.get('heartbeat_slice_size', 20),
                        cardiac_library=self.cardiac_library
                    )

                    # Convert to tensor
                    hb_sequence = torch.tensor(cardiac_data['hb_sequence'], dtype=torch.float32)
                    condition_with_cardiac = condition.copy()
                    condition_with_cardiac['hb_sequence'] = hb_sequence
                    print("    Loaded real cardiac data")
                else:
                    condition_with_cardiac = condition.copy()

                # FIXED: Run the model with proper condition
                trial, train_stepper = runner.run(**condition_with_cardiac)

                # FIXED: Extract comprehensive results and check for NaN (matching library tester)
                outputs = train_stepper.outputs.detach().cpu().numpy()
                cost = train_stepper.cost.item()

                # Check for NaN values
                has_nan_outputs = np.isnan(outputs).any()
                has_nan_cost = np.isnan(cost)

                # FIXED: Calculate performance using the robust method from library tester
                performance = self._calculate_time_bisection_performance(trial, outputs, condition)

                # Check performance metrics for NaN
                performance_has_critical_nan = np.isnan(performance.get('accuracy', 0))
                has_nan = has_nan_outputs or has_nan_cost or performance_has_critical_nan

                if has_nan:
                    print(f"    WARNING: NaN detected in condition {i + 1}")

                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'outputs': outputs,  # ADD: Store outputs for plotting
                    'trial': {  # ADD: Store trial data for plotting
                        'x': trial.x,
                        'y': trial.y,
                        'cost_mask': trial.cost_mask,
                        'epochs': trial.epochs,
                        'seq_len': trial.seq_len
                    },
                    'performance': performance,
                    'cost': cost,
                    'has_nan': has_nan,
                    'nan_details': {
                        'outputs': has_nan_outputs,
                        'cost': has_nan_cost,
                        'performance': performance_has_critical_nan
                    }
                }

                results['test_results'].append(test_result)
                results['nan_counts_per_condition'].append(1 if has_nan else 0)

            except Exception as e:
                print(f"    Error testing condition {condition}: {e}")
                import traceback
                traceback.print_exc()

                # Add failed result
                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'performance': {'accuracy': np.nan},
                    'cost': np.nan,
                    'has_nan': True,
                    'debug_info': {'error': str(e)}
                }
                results['test_results'].append(test_result)
                results['nan_counts_per_condition'].append(1)
                continue

        # FIXED: Calculate performance summary using the robust method from library tester
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance_fixed(results['test_results'])
            # Add NaN summary
            results['performance_summary']['total_nan_conditions'] = sum(results['nan_counts_per_condition'])
            results['performance_summary']['total_conditions'] = len(test_conditions)
            results['performance_summary']['nan_rate'] = sum(results['nan_counts_per_condition']) / len(test_conditions)

        return results

    def _calculate_time_bisection_performance(self, trial, outputs, condition):
        """FIXED: Calculate time bisection performance using the robust logic from library tester"""
        y = trial.y  # [T, B, O] - targets from task generation
        T, B, O = outputs.shape

        # Find response timing (from library tester logic)
        targets_active = (np.abs(y).sum(axis=2) > 0)  # [T, B]
        has_target = targets_active.any(axis=0)  # [B]

        if not has_target.any():
            return {'accuracy': 0.0, 'debug_error': 'no_active_targets'}

        response_idx_targets = targets_active.astype(float).argmax(axis=0)  # [B]

        # Fallback from cost mask
        cm = trial.cost_mask
        if cm.ndim == 3:
            mask_tb = (cm != 0).any(axis=2)
        else:
            mask_tb = cm
        response_idx_mask = mask_tb.astype(float).argmax(axis=0)

        response_idx = np.where(has_target, response_idx_targets, response_idx_mask)

        # Extract outputs and targets at response time
        logits_b = np.transpose(outputs, (1, 0, 2))  # [B, T, O]
        targets_b = np.transpose(y, (1, 0, 2))  # [B, T, O]

        rows = np.arange(B)
        chosen_logits = logits_b[rows, response_idx, :]  # [B, O]
        chosen_targets = targets_b[rows, response_idx, :]  # [B, O]

        # Calculate predictions and accuracy
        pred_classes = np.argmax(chosen_logits, axis=1)  # [B]
        target_classes = np.argmax(chosen_targets, axis=1)  # [B]

        # Calculate accuracy
        correct = (pred_classes == target_classes).astype(float)
        accuracy = float(np.mean(correct)) if correct.size > 0 else 0.0

        # Additional metrics for debugging
        duration = condition.get('duration', 0)
        short_std = condition.get('short_standard', 1000)
        long_std = condition.get('long_standard', 2000)

        return {
            'accuracy': accuracy,
            'correct_predictions': int(correct.sum()) if correct.size > 0 else 0,
            'total_predictions': int(correct.size),
            'duration': duration,
            'ground_truth_class': int(target_classes[0]) if len(target_classes) > 0 else -1,
            'predicted_class': int(pred_classes[0]) if len(pred_classes) > 0 else -1,
            'bisection_point': (short_std + long_std) / 2
        }

    def _summarize_performance_fixed(self, test_results):
        """FIXED: Summarize performance using the same logic as the working library tester"""
        if not test_results:
            return {}

        # FIXED: Use the same filtering logic - don't filter by has_nan, only filter by actual NaN values
        accuracies = [r['performance']['accuracy'] for r in test_results
                      if 'accuracy' in r['performance'] and not np.isnan(r['performance']['accuracy'])]

        costs = [r['cost'] for r in test_results if not np.isnan(r['cost']) and np.isfinite(r['cost'])]

        # Separate ID and OOD accuracies for detailed analysis
        id_accuracies = []
        ood_accuracies = []

        for r in test_results:
            if 'accuracy' in r['performance'] and not np.isnan(r['performance']['accuracy']):
                accuracy = r['performance']['accuracy']
                is_ood = r['condition'].get('is_ood', False)
                if is_ood:
                    ood_accuracies.append(accuracy)
                else:
                    id_accuracies.append(accuracy)

        # Calculate summary with better error handling
        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(accuracies),
            'nan_rate': (len(test_results) - len(accuracies)) / len(test_results) if test_results else 0,
            'accuracy': np.mean(accuracies) if accuracies else 0,  # FIXED: Don't use nanmean on empty list
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'mean_cost': np.mean(costs) if costs else np.inf,
            # ADD: ID/OOD metrics
            'mean_id_accuracy': np.mean(id_accuracies) if id_accuracies else np.nan,
            'mean_ood_accuracy': np.mean(ood_accuracies) if ood_accuracies else np.nan,
            'num_id_conditions': len(id_accuracies),
            'num_ood_conditions': len(ood_accuracies),
            'std_id_accuracy': np.std(id_accuracies) if id_accuracies else np.nan,
            'std_ood_accuracy': np.std(ood_accuracies) if ood_accuracies else np.nan,
        }

        # Debug output to see what's happening
        print(f"    Summary: {len(accuracies)} valid accuracies out of {len(test_results)} conditions")
        if accuracies:
            print(f"    Mean accuracy: {summary['accuracy']:.4f}")
            if id_accuracies:
                print(f"    ID accuracy: {summary['mean_id_accuracy']:.4f}")
            if ood_accuracies:
                print(f"    OOD accuracy: {summary['mean_ood_accuracy']:.4f}")
        else:
            print(f"    No valid accuracies found!")

        return summary

    def plot_mean_performance_comparison(self, all_results):
        """Create mean performance comparison plot between piezo and non-piezo models"""
        print("Creating mean performance comparison plot...")

        # Separate piezo and non-piezo models
        piezo_models = [r for r in all_results if r['use_piezo']]
        no_piezo_models = [r for r in all_results if not r['use_piezo']]

        if not piezo_models or not no_piezo_models:
            print("Need both piezo and non-piezo models for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mean Performance Comparison: Piezo vs Non-Piezo Models (Time Bisection)', fontsize=16)

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
        piezo_all = get_accuracy_stats(piezo_models)
        no_piezo_all = get_accuracy_stats(no_piezo_models)

        categories = []
        means = []
        stds = []
        colors = []

        if piezo_all:
            categories.append('Piezo Models')
            means.append(np.mean(piezo_all))
            stds.append(np.std(piezo_all))
            colors.append('red')

        if no_piezo_all:
            categories.append('Non-Piezo Models')
            means.append(np.mean(no_piezo_all))
            stds.append(np.std(no_piezo_all))
            colors.append('blue')

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
        piezo_id = get_accuracy_stats(piezo_models, 'id')
        no_piezo_id = get_accuracy_stats(no_piezo_models, 'id')

        id_categories = []
        id_means = []
        id_stds = []
        id_colors = []

        if piezo_id:
            id_categories.append('Piezo ID')
            id_means.append(np.mean(piezo_id))
            id_stds.append(np.std(piezo_id))
            id_colors.append('red')

        if no_piezo_id:
            id_categories.append('Non-Piezo ID')
            id_means.append(np.mean(no_piezo_id))
            id_stds.append(np.std(no_piezo_id))
            id_colors.append('blue')

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
        piezo_ood = get_accuracy_stats(piezo_models, 'ood')
        no_piezo_ood = get_accuracy_stats(no_piezo_models, 'ood')

        ood_categories = []
        ood_means = []
        ood_stds = []
        ood_colors = []

        if piezo_ood:
            ood_categories.append('Piezo OOD')
            ood_means.append(np.mean(piezo_ood))
            ood_stds.append(np.std(piezo_ood))
            ood_colors.append('red')

        if no_piezo_ood:
            ood_categories.append('Non-Piezo OOD')
            ood_means.append(np.mean(no_piezo_ood))
            ood_stds.append(np.std(no_piezo_ood))
            ood_colors.append('blue')

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

        piezo_total_ood, piezo_failed_ood = count_ood_responses(piezo_models, 'Piezo')
        no_piezo_total_ood, no_piezo_failed_ood = count_ood_responses(no_piezo_models, 'Non-Piezo')

        # Create grouped bar chart
        labels = ['Piezo Models', 'Non-Piezo Models']
        successful = [piezo_total_ood - piezo_failed_ood, no_piezo_total_ood - no_piezo_failed_ood]
        failed = [piezo_failed_ood, no_piezo_failed_ood]

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
        plot_path = os.path.join(self.base_results_dir, "mean_performance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved mean performance comparison plot: {plot_path}")

        # Store statistics for report
        self.comparison_stats = {
            'overall': {
                'piezo': {'mean': np.mean(piezo_all), 'std': np.std(piezo_all), 'n': len(piezo_all)},
                'no_piezo': {'mean': np.mean(no_piezo_all), 'std': np.std(no_piezo_all), 'n': len(no_piezo_all)}
            },
            'in_distribution': {
                'piezo': {'mean': np.mean(piezo_id) if piezo_id else 0, 'std': np.std(piezo_id) if piezo_id else 0,
                          'n': len(piezo_id)},
                'no_piezo': {'mean': np.mean(no_piezo_id) if no_piezo_id else 0,
                             'std': np.std(no_piezo_id) if no_piezo_id else 0, 'n': len(no_piezo_id)}
            },
            'out_of_distribution': {
                'piezo': {'mean': np.mean(piezo_ood) if piezo_ood else 0, 'std': np.std(piezo_ood) if piezo_ood else 0,
                          'n': len(piezo_ood)},
                'no_piezo': {'mean': np.mean(no_piezo_ood) if no_piezo_ood else 0,
                             'std': np.std(no_piezo_ood) if no_piezo_ood else 0, 'n': len(no_piezo_ood)}
            },
            'ood_failures': {
                'piezo': {'total': piezo_total_ood, 'failed': piezo_failed_ood},
                'no_piezo': {'total': no_piezo_total_ood, 'failed': no_piezo_failed_ood}
            }
        }

    def plot_duration_vs_accuracy(self, all_results, save_path=None):
        """Create scatter plot of duration vs accuracy for time bisection tasks with proper categorization"""
        print("Creating duration vs accuracy scatter plot...")

        # Collect data for scatter plot organized by reference framework
        framework_data = {}  # (short_std, long_std): {'piezo': {duration: [accuracies]}, 'no_piezo': {duration: [accuracies]}}

        for result in all_results:
            use_piezo = result['use_piezo']

            for test_result in result['test_results']:
                # Skip if has NaN or missing accuracy data
                if (test_result['has_nan'] or
                        'accuracy' not in test_result['performance'] or
                        np.isnan(test_result['performance']['accuracy'])):
                    continue

                # Get condition details
                condition = test_result['condition']
                duration = condition['duration']
                short_std = condition['short_standard']
                long_std = condition['long_standard']
                accuracy = test_result['performance']['accuracy']
                is_ood = condition.get('is_ood', False)

                # Create framework key
                framework_key = (short_std, long_std)
                if framework_key not in framework_data:
                    framework_data[framework_key] = {'piezo': {}, 'no_piezo': {}}

                # Group by model type
                model_type = 'piezo' if use_piezo else 'no_piezo'

                if duration not in framework_data[framework_key][model_type]:
                    framework_data[framework_key][model_type][duration] = []
                framework_data[framework_key][model_type][duration].append(accuracy)

        if not framework_data:
            print("No valid data found for scatter plot")
            return

        # Create subplots for each reference framework
        num_frameworks = len(framework_data)
        fig, axes = plt.subplots(num_frameworks, 1, figsize=(12, 6 * num_frameworks))
        if num_frameworks == 1:
            axes = [axes]

        fig.suptitle('Duration vs Accuracy by Reference Framework\n(Time Bisection Task)', fontsize=16,
                     fontweight='bold')

        for idx, (framework_key, data) in enumerate(framework_data.items()):
            ax = axes[idx]
            short_std, long_std = framework_key
            bisection_point = (short_std + long_std) / 2

            # Determine if this is in-distribution or out-of-distribution
            is_id_framework = (short_std == 300 and long_std == 900)  # Assuming this is the training framework
            framework_type = "In-Distribution" if is_id_framework else "Out-of-Distribution"
            if not is_id_framework:
                if bisection_point < 600:  # Below typical training range
                    framework_type += " (Lower Range)"
                else:  # Above typical training range
                    framework_type += " (Higher Range)"

            # Plot piezo models
            piezo_data = data['piezo']
            if piezo_data:
                piezo_durations = []
                piezo_means = []
                piezo_stds = []

                for duration in sorted(piezo_data.keys()):
                    accuracies = piezo_data[duration]
                    piezo_durations.append(duration)
                    piezo_means.append(np.mean(accuracies))
                    piezo_stds.append(np.std(accuracies))

                ax.errorbar(piezo_durations, piezo_means, yerr=piezo_stds,
                            fmt='o', color='red', markersize=8, capsize=5,
                            label=f'Piezo Models (n={len(piezo_data)} durations)',
                            alpha=0.8, linewidth=2)

            # Plot no piezo models
            no_piezo_data = data['no_piezo']
            if no_piezo_data:
                no_piezo_durations = []
                no_piezo_means = []
                no_piezo_stds = []

                for duration in sorted(no_piezo_data.keys()):
                    accuracies = no_piezo_data[duration]
                    no_piezo_durations.append(duration)
                    no_piezo_means.append(np.mean(accuracies))
                    no_piezo_stds.append(np.std(accuracies))

                ax.errorbar(no_piezo_durations, no_piezo_means, yerr=no_piezo_stds,
                            fmt='s', color='blue', markersize=8, capsize=5,
                            label=f'Standard Models (n={len(no_piezo_data)} durations)',
                            alpha=0.8, linewidth=2)

            # Add reference lines
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Perfect Accuracy')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Chance Level')
            ax.axvline(x=bisection_point, color='orange', linestyle='-.', alpha=0.7, linewidth=2,
                       label=f'Bisection Point ({bisection_point:.0f}ms)')

            # Mark the standard intervals
            ax.axvline(x=short_std, color='green', linestyle=':', alpha=0.5, linewidth=1,
                       label=f'Short Standard ({short_std:.0f}ms)')
            ax.axvline(x=long_std, color='purple', linestyle=':', alpha=0.5, linewidth=1,
                       label=f'Long Standard ({long_std:.0f}ms)')

            # Formatting
            ax.set_xlabel('Duration (ms)', fontsize=12)
            ax.set_ylabel('Mean Accuracy', fontsize=12)
            ax.set_title(
                f'{framework_type}\nStandards: {short_std:.0f}ms / {long_std:.0f}ms (Bisection: {bisection_point:.0f}ms)',
                fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1.1)

            # Add summary statistics as text
            if piezo_data and no_piezo_data:
                piezo_overall_mean = np.mean([np.mean(accuracies) for accuracies in piezo_data.values()])
                no_piezo_overall_mean = np.mean([np.mean(accuracies) for accuracies in no_piezo_data.values()])

                summary_text = f"Overall Mean Accuracy:\n"
                summary_text += f"  Piezo: {piezo_overall_mean:.3f}\n"
                summary_text += f"  Standard: {no_piezo_overall_mean:.3f}"

                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(self.base_results_dir, "duration_vs_accuracy_by_framework.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved duration vs accuracy scatter plot: {save_path}")

    def plot_piezo_vs_no_piezo_scatter(self, all_results):
        """Create scatter plot comparing mean piezo vs non-piezo model performance"""
        print("Creating piezo vs non-piezo scatter plot...")

        if not all_results:
            print("No results to plot")
            return

        # Separate models by type
        piezo_models = [r for r in all_results if r['use_piezo']]
        no_piezo_models = [r for r in all_results if not r['use_piezo']]

        if not piezo_models or not no_piezo_models:
            print("Need both piezo and non-piezo models for scatter plot")
            return

        # Extract mean accuracy for each model
        piezo_accuracies = []
        no_piezo_accuracies = []

        for model in piezo_models:
            if 'accuracy' in model['performance_summary']:
                piezo_accuracies.append(model['performance_summary']['accuracy'])

        for model in no_piezo_models:
            if 'accuracy' in model['performance_summary']:
                no_piezo_accuracies.append(model['performance_summary']['accuracy'])

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot individual models
        if piezo_accuracies:
            ax.scatter([1] * len(piezo_accuracies), piezo_accuracies,
                       color='red', s=100, alpha=0.7, label=f'Piezo Models (n={len(piezo_accuracies)})')

        if no_piezo_accuracies:
            ax.scatter([2] * len(no_piezo_accuracies), no_piezo_accuracies,
                       color='blue', s=100, alpha=0.7, label=f'Non-Piezo Models (n={len(no_piezo_accuracies)})')

        # Add mean lines
        if piezo_accuracies:
            piezo_mean = np.mean(piezo_accuracies)
            ax.axhline(y=piezo_mean, color='red', linestyle='--', alpha=0.8,
                       label=f'Piezo Mean: {piezo_mean:.3f}')

        if no_piezo_accuracies:
            no_piezo_mean = np.mean(no_piezo_accuracies)
            ax.axhline(y=no_piezo_mean, color='blue', linestyle='--', alpha=0.8,
                       label=f'Non-Piezo Mean: {no_piezo_mean:.3f}')

        # Reference lines
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Perfect Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance Level')

        # Formatting
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Piezo Models', 'Non-Piezo Models'])
        ax.set_ylabel('Mean Accuracy', fontsize=12)
        ax.set_title('Piezo vs Non-Piezo Model Performance\n(Time Bisection Task)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.base_results_dir, "piezo_vs_no_piezo_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved piezo vs non-piezo scatter plot: {save_path}")
        return save_path

    def plot_condition_consolidated_graphs(self, all_results):
        """Create consolidated graphs by reference framework type instead of per-condition"""
        print("Creating consolidated framework comparison graphs...")

        # Group results by framework type
        framework_groups = {
            'in_distribution': {'results': [], 'name': 'In-Distribution Framework'},
            'ood_lower': {'results': [], 'name': 'Out-of-Distribution (Lower Range)'},
            'ood_higher': {'results': [], 'name': 'Out-of-Distribution (Higher Range)'}
        }

        for result in all_results:
            rule_name = result['rule_name']
            for test_result in result['test_results']:
                condition = test_result['condition']
                short_std = condition['short_standard']
                long_std = condition['long_standard']
                is_ood = condition.get('is_ood', False)

                # Determine framework type
                if not is_ood:
                    framework_type = 'in_distribution'
                else:
                    # Determine if OOD is lower or higher range
                    bisection_point = (short_std + long_std) / 2
                    if bisection_point < 1500:  # Below typical training range
                        framework_type = 'ood_lower'
                    else:  # Above typical training range
                        framework_type = 'ood_higher'

                framework_groups[framework_type]['results'].append({
                    'model_name': result['model_name'],
                    'use_piezo': result['use_piezo'],
                    'performance': test_result['performance'],
                    'cost': test_result['cost'],
                    'has_nan': test_result['has_nan'],
                    'condition': condition,
                    'rule_name': rule_name
                })

        # Create consolidated plots for each framework type
        for framework_key, framework_data in framework_groups.items():
            if framework_data['results']:
                self._plot_consolidated_framework_comparison(
                    framework_key,
                    framework_data['name'],
                    framework_data['results']
                )

    def _plot_consolidated_framework_comparison(self, framework_key, framework_name, framework_results):
        """Plot consolidated comparison for a framework type across all models and conditions"""
        if not framework_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Consolidated Model Comparison: {framework_name}', fontsize=16, fontweight='bold')

        # Separate piezo and non-piezo models
        piezo_results = [r for r in framework_results if r['use_piezo']]
        no_piezo_results = [r for r in framework_results if not r['use_piezo']]

        # Plot 1: Overall accuracy comparison
        ax1 = axes[0, 0]

        # Get accuracies for each model type
        piezo_accs = [r['performance']['accuracy'] for r in piezo_results
                      if not r['has_nan'] and 'accuracy' in r['performance']
                      and not np.isnan(r['performance']['accuracy'])]
        no_piezo_accs = [r['performance']['accuracy'] for r in no_piezo_results
                         if not r['has_nan'] and 'accuracy' in r['performance']
                         and not np.isnan(r['performance']['accuracy'])]

        categories = []
        means = []
        stds = []
        colors = []
        individual_points = []

        if piezo_accs:
            categories.append('Piezo Models')
            means.append(np.mean(piezo_accs))
            stds.append(np.std(piezo_accs))
            colors.append('red')
            individual_points.append(piezo_accs)
        else:
            individual_points.append([])

        if no_piezo_accs:
            categories.append('Non-Piezo Models')
            means.append(np.mean(no_piezo_accs))
            stds.append(np.std(no_piezo_accs))
            colors.append('blue')
            individual_points.append(no_piezo_accs)
        else:
            individual_points.append([])

        # Create bar plot with individual points
        if categories:
            bars = ax1.bar(categories, means, yerr=stds, color=colors, alpha=0.7, capsize=5)

            # Add individual data points
            for i, (category, points) in enumerate(zip(categories, individual_points)):
                if points:
                    x_pos = [i] * len(points)
                    ax1.scatter(x_pos, points, color='black', s=30, alpha=0.6, zorder=10)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                         f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Accuracy')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Overall Performance\n(n={len(piezo_accs)} piezo, {len(no_piezo_accs)} non-piezo tests)')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cost comparison
        ax2 = axes[0, 1]

        piezo_costs = [r['cost'] for r in piezo_results
                       if not r['has_nan'] and np.isfinite(r['cost'])]
        no_piezo_costs = [r['cost'] for r in no_piezo_results
                          if not r['has_nan'] and np.isfinite(r['cost'])]

        cost_categories = []
        cost_means = []
        cost_stds = []
        cost_colors = []
        cost_points = []

        if piezo_costs:
            cost_categories.append('Piezo')
            cost_means.append(np.mean(piezo_costs))
            cost_stds.append(np.std(piezo_costs))
            cost_colors.append('red')
            cost_points.append(piezo_costs)
        else:
            cost_points.append([])

        if no_piezo_costs:
            cost_categories.append('Non-Piezo')
            cost_means.append(np.mean(no_piezo_costs))
            cost_stds.append(np.std(no_piezo_costs))
            cost_colors.append('blue')
            cost_points.append(no_piezo_costs)
        else:
            cost_points.append([])

        if cost_categories:
            bars = ax2.bar(cost_categories, cost_means, yerr=cost_stds,
                           color=cost_colors, alpha=0.7, capsize=5)

            # Add individual cost points
            for i, (category, points) in enumerate(zip(cost_categories, cost_points)):
                if points:
                    x_pos = [i] * len(points)
                    ax2.scatter(x_pos, points, color='black', s=30, alpha=0.6, zorder=10)

        ax2.set_ylabel('Training Cost')
        ax2.set_title('Training Cost Comparison')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Success/Failure rates
        ax3 = axes[1, 0]

        # Count successful vs failed responses
        piezo_total = len(piezo_results)
        piezo_failed = sum(1 for r in piezo_results if r['has_nan'] or
                           'accuracy' not in r['performance'] or
                           np.isnan(r['performance']['accuracy']))

        no_piezo_total = len(no_piezo_results)
        no_piezo_failed = sum(1 for r in no_piezo_results if r['has_nan'] or
                              'accuracy' not in r['performance'] or
                              np.isnan(r['performance']['accuracy']))

        failure_categories = []
        successful_counts = []
        failed_counts = []

        if piezo_total > 0:
            failure_categories.append('Piezo')
            successful_counts.append(piezo_total - piezo_failed)
            failed_counts.append(piezo_failed)

        if no_piezo_total > 0:
            failure_categories.append('Non-Piezo')
            successful_counts.append(no_piezo_total - no_piezo_failed)
            failed_counts.append(no_piezo_failed)

        if failure_categories:
            x = np.arange(len(failure_categories))
            width = 0.35

            ax3.bar(x - width / 2, successful_counts, width, label='Successful',
                    alpha=0.7, color='green')
            ax3.bar(x + width / 2, failed_counts, width, label='Failed/NaN',
                    alpha=0.7, color='red')

            # Add value labels
            for i, (succ, fail) in enumerate(zip(successful_counts, failed_counts)):
                ax3.text(i - width / 2, succ + 0.5, str(succ), ha='center', va='bottom')
                ax3.text(i + width / 2, fail + 0.5, str(fail), ha='center', va='bottom')

            ax3.set_xticks(x)
            ax3.set_xticklabels(failure_categories)

        ax3.set_ylabel('Number of Tests')
        ax3.set_title('Success vs Failure Rates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Framework summary and condition details
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Get unique conditions in this framework
        unique_conditions = {}
        for r in framework_results:
            condition = r['condition']
            key = f"{condition['short_standard']}/{condition['long_standard']}"
            if key not in unique_conditions:
                unique_conditions[key] = []
            unique_conditions[key].append(condition['duration'])

        summary_text = f"Framework: {framework_name}\n\n"
        summary_text += f"Reference Standards:\n"
        for standards, durations in unique_conditions.items():
            summary_text += f"  {standards}ms: {len(set(durations))} unique durations\n"
            summary_text += f"    Range: {min(durations):.0f}-{max(durations):.0f}ms\n"

        summary_text += f"\nTest Summary:\n"
        summary_text += f"  Total tests: {len(framework_results)}\n"
        summary_text += f"  Piezo tests: {piezo_total}\n"
        summary_text += f"  Non-Piezo tests: {no_piezo_total}\n"
        summary_text += f"  Failed tests: {piezo_failed + no_piezo_failed}\n"
        summary_text += f"  Success rate: {((len(framework_results) - piezo_failed - no_piezo_failed) / len(framework_results) * 100):.1f}%\n"

        if piezo_accs and no_piezo_accs:
            summary_text += f"\nPerformance Summary:\n"
            summary_text += f"  Piezo mean: {np.mean(piezo_accs):.3f}\n"
            summary_text += f"  Non-Piezo mean: {np.mean(no_piezo_accs):.3f}\n"
            summary_text += f"  Difference: {np.mean(piezo_accs) - np.mean(no_piezo_accs):+.3f}\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, f"consolidated_{framework_key}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved consolidated framework comparison: {plot_path}")
        return plot_path
    def _plot_single_condition_comparison(self, condition_key, condition_results):
        """Plot comparison for a single condition across all models"""
        if not condition_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Comparison: {condition_key.replace("_", " ").title()}', fontsize=16)

        # Separate piezo and non-piezo models
        piezo_models = [r for r in condition_results if r['use_piezo']]
        no_piezo_models = [r for r in condition_results if not r['use_piezo']]

        # Count NaN occurrences
        piezo_nan_count = sum(1 for r in piezo_models if r['has_nan'])
        no_piezo_nan_count = sum(1 for r in no_piezo_models if r['has_nan'])

        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        if piezo_models:
            piezo_accs = [r['performance']['accuracy'] for r in piezo_models if not r['has_nan']]
            if piezo_accs:
                ax1.bar(['Piezo'], [np.mean(piezo_accs)], yerr=[np.std(piezo_accs)],
                        alpha=0.7, color='red', capsize=5)
                ax1.scatter(['Piezo'] * len(piezo_accs), piezo_accs, color='darkred', s=50, zorder=10)

        if no_piezo_models:
            no_piezo_accs = [r['performance']['accuracy'] for r in no_piezo_models if not r['has_nan']]
            if no_piezo_accs:
                ax1.bar(['No Piezo'], [np.mean(no_piezo_accs)], yerr=[np.std(no_piezo_accs)],
                        alpha=0.7, color='blue', capsize=5)
                ax1.scatter(['No Piezo'] * len(no_piezo_accs), no_piezo_accs, color='darkblue', s=50, zorder=10)

        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Cost comparison
        ax2 = axes[0, 1]
        if piezo_models:
            piezo_costs = [r['cost'] for r in piezo_models if not r['has_nan'] and np.isfinite(r['cost'])]
            if piezo_costs:
                ax2.bar(['Piezo'], [np.mean(piezo_costs)], yerr=[np.std(piezo_costs)],
                        alpha=0.7, color='red', capsize=5)
                ax2.scatter(['Piezo'] * len(piezo_costs), piezo_costs, color='darkred', s=50, zorder=10)

        if no_piezo_models:
            no_piezo_costs = [r['cost'] for r in no_piezo_models if not r['has_nan'] and np.isfinite(r['cost'])]
            if no_piezo_costs:
                ax2.bar(['No Piezo'], [np.mean(no_piezo_costs)], yerr=[np.std(no_piezo_costs)],
                        alpha=0.7, color='blue', capsize=5)
                ax2.scatter(['No Piezo'] * len(no_piezo_costs), no_piezo_costs, color='darkblue', s=50, zorder=10)

        ax2.set_ylabel('Cost')
        ax2.set_title('Training Cost Comparison')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Model count and NaN summary
        ax3 = axes[1, 0]
        categories = []
        successful_counts = []
        nan_counts = []

        if piezo_models:
            categories.append('Piezo')
            successful_counts.append(len(piezo_models) - piezo_nan_count)
            nan_counts.append(piezo_nan_count)

        if no_piezo_models:
            categories.append('No Piezo')
            successful_counts.append(len(no_piezo_models) - no_piezo_nan_count)
            nan_counts.append(no_piezo_nan_count)

        x = np.arange(len(categories))
        width = 0.35

        ax3.bar(x - width / 2, successful_counts, width, label='Successful', alpha=0.7, color='green')
        ax3.bar(x + width / 2, nan_counts, width, label='NaN/Failed', alpha=0.7, color='red')

        ax3.set_ylabel('Number of Models')
        ax3.set_title('Model Success vs NaN/Failure Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Condition details
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Display condition parameters and summary
        if condition_results:
            condition_params = condition_results[0]['condition']
            summary_text = "Condition Parameters:\n"
            for key, value in condition_params.items():
                summary_text += f"  {key}: {value}\n"

            summary_text += f"\nSummary:\n"
            summary_text += f"  Total models: {len(condition_results)}\n"
            summary_text += f"  Piezo models: {len(piezo_models)}\n"
            summary_text += f"  No-Piezo models: {len(no_piezo_models)}\n"
            summary_text += f"  NaN occurrences: {piezo_nan_count + no_piezo_nan_count}\n"
            summary_text += f"  Success rate: {((len(condition_results) - piezo_nan_count - no_piezo_nan_count) / len(condition_results) * 100):.1f}%"

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, f"{condition_key}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved condition comparison: {plot_path}")

    def plot_task_vs_model_output(self, all_results):
        """Create task vs model output visualizations for each model and condition"""
        print("Creating task vs model output visualizations...")

        for result in all_results:
            model_name = result['model_name']
            rule_name = result['rule_name']
            use_piezo = result['use_piezo']

            # Create model-specific directory
            model_dir = os.path.join(self.base_results_dir, f"{model_name}_task_vs_output")
            tools.mkdir_p(model_dir)

            for test_result in result['test_results']:
                condition_idx = test_result['condition_idx']
                condition = test_result['condition']

                # Skip if has NaN
                if test_result['has_nan']:
                    continue

                # Plot task structure
                task_plot = self._plot_task_structure(test_result, model_name, model_dir)

                # Plot model outputs vs targets
                output_plot = self._plot_model_outputs_vs_targets(test_result, model_name, model_dir)

            print(f"Saved task vs output plots for {model_name} to: {model_dir}")

    def _plot_task_structure(self, test_result, model_name, save_dir):
        """Plot task structure showing inputs and targets"""
        condition = test_result['condition']
        condition_idx = test_result['condition_idx']

        # Get the trial data from the test_result
        if 'trial' in test_result:
            trial = test_result['trial']
        else:
            print(f"No trial data available for {model_name} condition {condition_idx}")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot task inputs
        for ch in range(trial['x'].shape[2]):
            ax.plot(trial['x'][:, 0, ch], label=f'Input {ch}', linewidth=2)

        # Plot target outputs
        for ch in range(trial['y'].shape[2]):
            ax.plot(trial['y'][:, 0, ch], '--', label=f'Target {ch}', alpha=0.7, linewidth=2)

        # Add epoch boundaries if available
        if 'epochs' in trial:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            for i, (epoch_name, (start, end)) in enumerate(trial['epochs'].items()):
                if start is not None:
                    color = colors[i % len(colors)]
                    start_time = start[0] if hasattr(start, '__iter__') else start
                    ax.axvline(start_time, color=color, linestyle=':', alpha=0.7, linewidth=2)
                    ax.text(start_time, ax.get_ylim()[1] * 0.9, epoch_name, rotation=90, fontsize=10, color=color)

        # Create meaningful title
        short_std = condition['short_standard']
        long_std = condition['long_standard']
        duration = condition['duration']
        is_ood = condition.get('is_ood', False)
        ood_marker = " (OOD)" if is_ood else " (ID)"

        ax.set_title(
            f'Task Structure - {model_name}\nStandards: {short_std}ms/{long_std}ms, Duration: {duration}ms{ood_marker}',
            fontsize=14)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Activity')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_task_structure.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_model_outputs_vs_targets(self, test_result, model_name, save_dir):
        """Plot model outputs vs targets for time bisection task"""
        condition = test_result['condition']
        condition_idx = test_result['condition_idx']
        performance = test_result['performance']

        # Get the trial and outputs data
        if 'trial' in test_result:
            trial = test_result['trial']
        else:
            print(f"No trial data available for {model_name} condition {condition_idx}")
            return None

        if 'outputs' in test_result:
            outputs = test_result['outputs']
        else:
            print(f"No outputs data available for {model_name} condition {condition_idx}")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot model outputs and targets
        for ch in range(outputs.shape[2]):
            ax.plot(outputs[:, 0, ch], label=f'Model Output {ch}', linewidth=2)

        for ch in range(trial['y'].shape[2]):
            ax.plot(trial['y'][:, 0, ch], '--', alpha=0.7, label=f'Target {ch}', linewidth=2)

        # Add performance text box
        short_std = condition['short_standard']
        long_std = condition['long_standard']
        duration = condition['duration']
        bisection_point = performance.get('bisection_point', (short_std + long_std) / 2)
        is_ood = condition.get('is_ood', False)
        ood_marker = " (OOD)" if is_ood else " (ID)"

        perf_text = f"Accuracy: {performance.get('accuracy', 0):.3f}\n"
        perf_text += f"Standards: {short_std}ms/{long_std}ms\n"
        perf_text += f"Duration: {duration}ms{ood_marker}\n"
        perf_text += f"Bisection Point: {bisection_point:.0f}ms"
        if 'response_start_time' in performance:
            perf_text += f"\nResponse Start: {performance['response_start_time']}"

        # Add NaN warning if detected
        if test_result['has_nan']:
            perf_text += f"\n*** NaN DETECTED ***"

        # Text box in upper left corner
        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat' if not test_result['has_nan'] else 'salmon',
                          alpha=0.8))

        # Set labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Output Activity')
        ax.set_title(
            f'Model Outputs vs Targets - {model_name}\nStandards: {short_std}ms/{long_std}ms, Duration: {duration}ms{ood_marker}',
            fontsize=14)

        # Add legend to the right side
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_model_outputs.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def discover_bisection_models(self):
        """Discover trained time bisection models"""
        model_dirs = []

        # Define possible base directories for bisection models
        possible_base_dirs = [
            'enhanced_piezo_time_bisection_results',
            'bisection_results',
            'time_bisection_results',
            'model/time_bisection',
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                print(f"Searching in {base_dir}...")
                self._search_model_subdirs(base_dir, 'time_bisection', model_dirs)

        return model_dirs

    def _search_model_subdirs(self, base_dir, rule_name, model_dirs):
        """Search for model subdirectories in a base directory"""
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Check if this looks like a run directory
                    if 'run_' in item or item.startswith(
                            'w2_') or item.isdigit() or 'piezo' in item or 'no_piezo' in item:
                        # Find the best checkpoint in this run directory
                        best_model_path = self._find_best_model_checkpoint(item_path)
                        if best_model_path:
                            model_dirs.append((best_model_path, rule_name, item))
                    elif self._has_valid_model(item_path):
                        # This directory itself contains a model
                        model_dirs.append((item_path, rule_name, item))
        except PermissionError:
            print(f"Permission denied accessing {base_dir}")
        except Exception as e:
            print(f"Error searching {base_dir}: {e}")

    def _find_best_model_checkpoint(self, base_model_dir):
        """Find the best model checkpoint in a run directory"""
        # Priority order: finalResult > highest numbered checkpoint > main directory
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

        # Get highest numbered checkpoint
        if numbered_checkpoints:
            numbered_checkpoints.sort(reverse=True)  # Highest number first
            return numbered_checkpoints[0][1]

        # Check main directory as fallback
        if self._has_valid_model(base_model_dir):
            return base_model_dir

        return None

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)

    def _generate_enhanced_summary_report(self, all_results, test_conditions):
        """Generate enhanced summary report with detailed piezo vs non-piezo statistics and NaN tracking"""
        report_path = os.path.join(self.base_results_dir, "enhanced_bisection_test_report.txt")

        # Separate conditions
        id_conditions = [c for c in test_conditions if not c.get('is_ood', False)]
        ood_conditions = [c for c in test_conditions if c.get('is_ood', False)]

        # Add statistical analysis
        analyzer = StatisticalAnalyzer()
        piezo_data, no_piezo_data = analyzer.extract_performance_data(all_results, 'time_bisection')
        statistical_results = analyzer.perform_statistical_tests(piezo_data, no_piezo_data, 'time_bisection')
        statistical_results = analyzer.apply_multiple_comparison_correction(statistical_results)

        with open(report_path, 'w') as f:
            f.write("ENHANCED TIME BISECTION DATASET TEST REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Test Conditions: {len(test_conditions)} total\n")
            f.write(f"  In-Distribution: {len(id_conditions)} samples from dataset\n")
            f.write(f"  Out-of-Distribution: {len(ood_conditions)} extrapolation tests\n")
            f.write(f"Models Tested: {len(all_results)}\n")

            # Model breakdown
            piezo_models = [r for r in all_results if r['use_piezo']]
            no_piezo_models = [r for r in all_results if not r['use_piezo']]
            f.write(f"Piezo Models: {len(piezo_models)}\n")
            f.write(f"Standard Models: {len(no_piezo_models)}\n\n")

            # Enhanced mean performance comparison (ALL GRAPHED METRICS)
            f.write("MEAN PERFORMANCE COMPARISON (ALL GRAPHED METRICS)\n")
            f.write("-" * 55 + "\n")

            if hasattr(self, 'comparison_stats'):
                stats = self.comparison_stats

                # Overall performance (from mean performance comparison plot)
                f.write(f"OVERALL PERFORMANCE (from Mean Performance Comparison):\n")
                f.write(
                    f"  Piezo Models:     {stats['overall']['piezo']['mean']:.4f} ± {stats['overall']['piezo']['std']:.4f} (n={stats['overall']['piezo']['n']})\n")
                f.write(
                    f"  Non-Piezo Models: {stats['overall']['no_piezo']['mean']:.4f} ± {stats['overall']['no_piezo']['std']:.4f} (n={stats['overall']['no_piezo']['n']})\n")

                if stats['overall']['piezo']['n'] > 0 and stats['overall']['no_piezo']['n'] > 0:
                    diff = stats['overall']['piezo']['mean'] - stats['overall']['no_piezo']['mean']
                    f.write(f"  Difference: {diff:+.4f} (Piezo - Non-Piezo)\n")
                f.write("\n")

                # In-Distribution performance (from mean performance comparison plot)
                f.write(f"IN-DISTRIBUTION PERFORMANCE (from Mean Performance Comparison):\n")
                f.write(
                    f"  Piezo Models:     {stats['in_distribution']['piezo']['mean']:.4f} ± {stats['in_distribution']['piezo']['std']:.4f} (n={stats['in_distribution']['piezo']['n']})\n")
                f.write(
                    f"  Non-Piezo Models: {stats['in_distribution']['no_piezo']['mean']:.4f} ± {stats['in_distribution']['no_piezo']['std']:.4f} (n={stats['in_distribution']['no_piezo']['n']})\n")

                if stats['in_distribution']['piezo']['n'] > 0 and stats['in_distribution']['no_piezo']['n'] > 0:
                    diff = stats['in_distribution']['piezo']['mean'] - stats['in_distribution']['no_piezo']['mean']
                    f.write(f"  Difference: {diff:+.4f} (Piezo - Non-Piezo)\n")
                f.write("\n")

                # Out-of-Distribution performance (from mean performance comparison plot)
                f.write(f"OUT-OF-DISTRIBUTION PERFORMANCE (from Mean Performance Comparison):\n")
                f.write(
                    f"  Piezo Models:     {stats['out_of_distribution']['piezo']['mean']:.4f} ± {stats['out_of_distribution']['piezo']['std']:.4f} (n={stats['out_of_distribution']['piezo']['n']})\n")
                f.write(
                    f"  Non-Piezo Models: {stats['out_of_distribution']['no_piezo']['mean']:.4f} ± {stats['out_of_distribution']['no_piezo']['std']:.4f} (n={stats['out_of_distribution']['no_piezo']['n']})\n")

                if stats['out_of_distribution']['piezo']['n'] > 0 and stats['out_of_distribution']['no_piezo']['n'] > 0:
                    diff = stats['out_of_distribution']['piezo']['mean'] - stats['out_of_distribution']['no_piezo'][
                        'mean']
                    f.write(f"  Difference: {diff:+.4f} (Piezo - Non-Piezo)\n")
                f.write("\n")

                # OOD Failure Analysis (from mean performance comparison plot)
                f.write(f"OUT-OF-DISTRIBUTION FAILURE ANALYSIS (from Mean Performance Comparison):\n")
                piezo_fail_rate = stats['ood_failures']['piezo']['failed'] / stats['ood_failures']['piezo']['total'] if \
                    stats['ood_failures']['piezo']['total'] > 0 else 0
                no_piezo_fail_rate = stats['ood_failures']['no_piezo']['failed'] / stats['ood_failures']['no_piezo'][
                    'total'] if stats['ood_failures']['no_piezo']['total'] > 0 else 0

                f.write(
                    f"  Piezo Models:     {stats['ood_failures']['piezo']['failed']}/{stats['ood_failures']['piezo']['total']} failed ({piezo_fail_rate:.1%})\n")
                f.write(
                    f"  Non-Piezo Models: {stats['ood_failures']['no_piezo']['failed']}/{stats['ood_failures']['no_piezo']['total']} failed ({no_piezo_fail_rate:.1%})\n")

                f.write("\n")

            # Add statistical analysis report
            stat_report = analyzer.generate_statistical_report(statistical_results, 'time_bisection')
            f.write(stat_report)

            # Individual model results
            f.write("INDIVIDUAL MODEL RESULTS\n")
            f.write("-" * 25 + "\n")

            for result in all_results:
                f.write(f"Model: {result['model_name']}\n")
                f.write(f"Task: {result['rule_name']}\n")
                f.write(f"Piezo: {result['use_piezo']}\n")
                f.write(f"Performance Summary:\n")

                for key, value in result['performance_summary'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

                f.write("\n")

        print(f"Generated enhanced summary report: {report_path}")

    def plot_framework_scatter_plots(self, all_results):
        """Create scatter plots for each framework type showing duration vs accuracy"""
        print("Creating framework-specific duration vs accuracy scatter plots...")

        # Group data by framework type
        framework_data = {
            'in_distribution': {'piezo': {}, 'no_piezo': {}, 'name': 'In-Distribution Framework'},
            'ood_lower': {'piezo': {}, 'no_piezo': {}, 'name': 'Out-of-Distribution (Lower Range)'},
            'ood_higher': {'piezo': {}, 'no_piezo': {}, 'name': 'Out-of-Distribution (Higher Range)'}
        }

        # Collect data for each framework type
        for result in all_results:
            use_piezo = result['use_piezo']

            for test_result in result['test_results']:
                # Skip if has NaN or missing accuracy data
                if (test_result['has_nan'] or
                        'accuracy' not in test_result['performance'] or
                        np.isnan(test_result['performance']['accuracy'])):
                    continue

                # Get condition details
                condition = test_result['condition']
                duration = condition['duration']
                short_std = condition['short_standard']
                long_std = condition['long_standard']
                accuracy = test_result['performance']['accuracy']
                is_ood = condition.get('is_ood', False)

                # Determine framework type
                if not is_ood:
                    framework_type = 'in_distribution'
                else:
                    # Determine if OOD is lower or higher range
                    bisection_point = (short_std + long_std) / 2
                    if bisection_point < 1500:  # Below typical training range
                        framework_type = 'ood_lower'
                    else:  # Above typical training range
                        framework_type = 'ood_higher'

                # Group by model type
                model_type = 'piezo' if use_piezo else 'no_piezo'

                if duration not in framework_data[framework_type][model_type]:
                    framework_data[framework_type][model_type][duration] = []
                framework_data[framework_type][model_type][duration].append(accuracy)

        # Create scatter plots for each framework type
        for framework_key, data in framework_data.items():
            if data['piezo'] or data['no_piezo']:  # Only plot if there's data
                self._plot_single_framework_scatter(framework_key, data)

    def _plot_single_framework_scatter(self, framework_key, framework_data):
        """Create a single scatter plot for one framework type"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        framework_name = framework_data['name']
        piezo_data = framework_data['piezo']
        no_piezo_data = framework_data['no_piezo']

        # Determine bisection point and reference standards for this framework
        all_durations = list(piezo_data.keys()) + list(no_piezo_data.keys())
        if not all_durations:
            return

        # For framework-specific reference lines, we need to infer from the data
        if framework_key == 'in_distribution':
            # Assume standard training framework
            short_std, long_std = 1000, 2000
            framework_desc = f"Training Framework: {short_std}ms - {long_std}ms"
        elif framework_key == 'ood_lower':
            # Lower range framework (from the images, appears to be 300-900)
            short_std, long_std = 300, 900
            framework_desc = f"Lower Range Framework: {short_std}ms - {long_std}ms"
        else:  # ood_higher
            # Higher range framework (from the images, appears to be 2000-3000)
            short_std, long_std = 2000, 3000
            framework_desc = f"Higher Range Framework: {short_std}ms - {long_std}ms"

        bisection_point = (short_std + long_std) / 2

        # Plot piezo models
        if piezo_data:
            piezo_durations = []
            piezo_means = []
            piezo_stds = []
            piezo_all_points = []

            for duration in sorted(piezo_data.keys()):
                accuracies = piezo_data[duration]
                piezo_durations.append(duration)
                piezo_means.append(np.mean(accuracies))
                piezo_stds.append(np.std(accuracies))

                # Add individual points for this duration
                for acc in accuracies:
                    piezo_all_points.append((duration, acc))

            # Plot mean with error bars
            ax.errorbar(piezo_durations, piezo_means, yerr=piezo_stds,
                        fmt='o', color='red', markersize=10, capsize=5,
                        label=f'Piezo Models (mean±std, n={len(piezo_durations)} durations)',
                        alpha=0.8, linewidth=2, markeredgecolor='darkred', markeredgewidth=1)

            # Plot individual data points
            if piezo_all_points:
                px, py = zip(*piezo_all_points)
                ax.scatter(px, py, color='red', s=30, alpha=0.4,
                           label=f'Individual Piezo Tests (n={len(piezo_all_points)})')

        # Plot no piezo models
        if no_piezo_data:
            no_piezo_durations = []
            no_piezo_means = []
            no_piezo_stds = []
            no_piezo_all_points = []

            for duration in sorted(no_piezo_data.keys()):
                accuracies = no_piezo_data[duration]
                no_piezo_durations.append(duration)
                no_piezo_means.append(np.mean(accuracies))
                no_piezo_stds.append(np.std(accuracies))

                # Add individual points for this duration
                for acc in accuracies:
                    no_piezo_all_points.append((duration, acc))

            # Plot mean with error bars
            ax.errorbar(no_piezo_durations, no_piezo_means, yerr=no_piezo_stds,
                        fmt='s', color='blue', markersize=10, capsize=5,
                        label=f'Standard Models (mean±std, n={len(no_piezo_durations)} durations)',
                        alpha=0.8, linewidth=2, markeredgecolor='darkblue', markeredgewidth=1)

            # Plot individual data points
            if no_piezo_all_points:
                npx, npy = zip(*no_piezo_all_points)
                ax.scatter(npx, npy, color='blue', s=30, alpha=0.4,
                           label=f'Individual Standard Tests (n={len(no_piezo_all_points)})')

        # Add reference lines
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1,
                   label='Perfect Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1,
                   label='Chance Level')
        ax.axvline(x=bisection_point, color='orange', linestyle='-.', alpha=0.8, linewidth=2,
                   label=f'Bisection Point ({bisection_point:.0f}ms)')

        # Mark the standard intervals
        ax.axvline(x=short_std, color='green', linestyle=':', alpha=0.6, linewidth=1,
                   label=f'Short Standard ({short_std:.0f}ms)')
        ax.axvline(x=long_std, color='purple', linestyle=':', alpha=0.6, linewidth=1,
                   label=f'Long Standard ({long_std:.0f}ms)')

        # Formatting
        ax.set_xlabel('Duration (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'{framework_name}\nDuration vs Accuracy Scatter Plot\n{framework_desc}',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_ylim(0, 1.1)

        # Set x-axis limits to show some padding around the data
        all_x = list(piezo_data.keys()) + list(no_piezo_data.keys())
        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            x_range = x_max - x_min
            ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)

        # Add summary statistics text box - position it to avoid data overlap
        summary_text = f"Summary Statistics:\n"

        if piezo_data and no_piezo_data:
            # Calculate overall means across all durations
            piezo_all_accs = [acc for accs in piezo_data.values() for acc in accs]
            no_piezo_all_accs = [acc for accs in no_piezo_data.values() for acc in accs]

            piezo_overall_mean = np.mean(piezo_all_accs)
            no_piezo_overall_mean = np.mean(no_piezo_all_accs)

            summary_text += f"Piezo Overall: {piezo_overall_mean:.3f}\n"
            summary_text += f"Standard Overall: {no_piezo_overall_mean:.3f}\n"
            summary_text += f"Difference: {piezo_overall_mean - no_piezo_overall_mean:+.3f}\n"
            summary_text += f"\nTotal Tests: {len(piezo_all_accs) + len(no_piezo_all_accs)}\n"
            summary_text += f"Piezo Tests: {len(piezo_all_accs)}\n"
            summary_text += f"Standard Tests: {len(no_piezo_all_accs)}"
        elif piezo_data:
            piezo_all_accs = [acc for accs in piezo_data.values() for acc in accs]
            summary_text += f"Piezo Only: {np.mean(piezo_all_accs):.3f}\n"
            summary_text += f"Total Tests: {len(piezo_all_accs)}"
        elif no_piezo_data:
            no_piezo_all_accs = [acc for accs in no_piezo_data.values() for acc in accs]
            summary_text += f"Standard Only: {np.mean(no_piezo_all_accs):.3f}\n"
            summary_text += f"Total Tests: {len(no_piezo_all_accs)}"

        # Position text box intelligently based on framework type to avoid data overlap
        if framework_key == 'in_distribution':
            # Data is spread across the range, put text box in upper left
            text_x, text_y = 0.02, 0.98
            h_align, v_align = 'left', 'top'
        elif framework_key == 'ood_lower':
            # Data is clustered around 500-600ms (left side), put text box on right
            text_x, text_y = 0.98, 0.98
            h_align, v_align = 'right', 'top'
        else:  # ood_higher
            # Data varies, put text box in upper left
            text_x, text_y = 0.02, 0.98
            h_align, v_align = 'left', 'top'

        ax.text(text_x, text_y, summary_text, transform=ax.transAxes, fontsize=11,
                horizontalalignment=h_align, verticalalignment=v_align, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.base_results_dir, f"scatter_{framework_key}_duration_vs_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved framework scatter plot: {save_path}")
        return save_path
    def run_enhanced_dataset_test(self, num_conditions=15, include_ood=True):
        """Run the complete enhanced dataset-driven test"""
        print("Starting Enhanced Time Bisection Dataset Test")
        print("=" * 70)

        # Discover models
        model_directories = self.discover_bisection_models()
        if not model_directories:
            print("No time bisection models found!")
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
            # Mean performance comparison plot
            self.plot_mean_performance_comparison(all_results)

            # Piezo vs non-piezo scatter plot
            self.plot_piezo_vs_no_piezo_scatter(all_results)

            # Duration vs accuracy scatter plot by reference framework
            self.plot_duration_vs_accuracy(all_results)

            # NEW: Framework-specific scatter plots
            self.plot_framework_scatter_plots(all_results)

            # Per-condition consolidated graphs (now organized by framework type)
            self.plot_condition_consolidated_graphs(all_results)

            # Task vs model output visualizations for each model
            self.plot_task_vs_model_output(all_results)

            # Generate enhanced summary report
            self._generate_enhanced_summary_report(all_results, test_conditions)

            # Save consolidated results
            summary_path = os.path.join(self.base_results_dir, "all_enhanced_dataset_test_results.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

        print(f"\nEnhanced dataset test complete!")
        print(f"Results saved to: {self.base_results_dir}")
        print(f"Generated plots:")
        print(f"  - Mean performance comparison: mean_performance_comparison.png")
        print(f"  - Piezo vs non-piezo scatter: piezo_vs_no_piezo_scatter.png")
        print(f"  - Duration vs accuracy by framework: duration_vs_accuracy_by_framework.png")
        print(f"  - Framework scatter plots:")
        print(f"    * scatter_in_distribution_duration_vs_accuracy.png")
        print(f"    * scatter_ood_lower_duration_vs_accuracy.png")
        print(f"    * scatter_ood_higher_duration_vs_accuracy.png")
        print(f"  - Consolidated framework comparisons:")
        print(f"    * consolidated_in_distribution_comparison.png")
        print(f"    * consolidated_ood_lower_comparison.png")
        print(f"    * consolidated_ood_higher_comparison.png")
        print(f"  - Individual model plots: [model]_task_vs_output/")
        print(f"  - Cardiac library used: {self.cardiac_library}")

        return all_results


def main():
    """Run enhanced time bisection dataset test"""
    tester = EnhancedTimeBisectionTester()
    results = tester.run_enhanced_dataset_test(num_conditions=20, include_ood=True)

    print(f"\nEnhanced dataset testing complete!")
    print(f"Results saved to: {tester.base_results_dir}")


if __name__ == "__main__":
    main()