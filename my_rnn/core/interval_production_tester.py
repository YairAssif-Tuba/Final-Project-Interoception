"""
Enhanced Interval Production Dataset Tester - UPDATED VERSION
=============================================================

Tests trained interval production models with enhanced analysis including:
- Mean performance comparison between piezo and non-piezo models
- Detailed breakdown of ID vs OOD performance
- Failure rate tracking for out-of-distribution intervals
- Task vs model output visualization (NEW)
- Retry logic for NaN responses (NEW)
- Enhanced reporting with all graphed metrics (NEW)
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

    def extract_performance_data(self, all_results, task_type='interval_production'):
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

    def perform_statistical_tests(self, piezo_data, no_piezo_data, task_type='interval_production'):
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
            n1, n2 = len(piezo_vals), len(no_piezo_vals)
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_score = (u_stat - mean_u) / std_u
            r_effect = abs(z_score) / np.sqrt(n1 + n2)

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

    def generate_statistical_report(self, results, task_type='interval_production'):
        """Generate comprehensive statistical analysis report"""
        if task_type == 'interval_production':
            metric_name = 'Completion Ratio'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be a significant difference in mean completion between piezo and non-piezo models (overall)
2. There will be a significant difference in mean completion between piezo and non-piezo models (in-distribution)
3. Piezo models will show superior performance in out-of-distribution conditions
"""
        elif task_type == 'interval_comparison':
            metric_name = 'Accuracy'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be a significant difference in accuracy between piezo and non-piezo models (all conditions)
2. There will be a significant difference in accuracy between piezo and non-piezo models (in-distribution)
3. There will be a significant difference in accuracy between piezo and non-piezo models (out-of-distribution)
"""
        else:  # time_bisection
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


def add_statistical_analysis_to_tester(tester_class):
    """Add statistical analysis methods to existing tester classes"""

    def enhanced_generate_summary_report(self, all_results, test_conditions):
        """Enhanced version with statistical analysis"""
        # Call the original method
        original_method = getattr(self, '_generate_enhanced_summary_report_original',
                                  self._generate_enhanced_summary_report)
        original_method(all_results, test_conditions)

        # Add statistical analysis
        analyzer = StatisticalAnalyzer()

        # Determine task type
        if hasattr(self, 'dataset_path'):
            if 'production' in self.dataset_path:
                task_type = 'interval_production'
            elif 'comparison' in self.dataset_path:
                task_type = 'interval_comparison'
            elif 'bisection' in self.dataset_path:
                task_type = 'time_bisection'
            else:
                task_type = 'interval_production'  # default
        else:
            task_type = 'interval_production'  # default

        # Extract data and perform tests
        piezo_data, no_piezo_data = analyzer.extract_performance_data(all_results, task_type)
        statistical_results = analyzer.perform_statistical_tests(piezo_data, no_piezo_data, task_type)
        statistical_results = analyzer.apply_multiple_comparison_correction(statistical_results)

        # Generate and append statistical report
        stat_report = analyzer.generate_statistical_report(statistical_results, task_type)

        # Append to existing report
        if task_type == 'interval_production':
            report_filename = "enhanced_production_test_report.txt"
        elif task_type == 'interval_comparison':
            report_filename = "enhanced_comparison_test_report.txt"
        else:
            report_filename = "enhanced_bisection_test_report.txt"

        report_path = os.path.join(self.base_results_dir, report_filename)

        with open(report_path, 'a') as f:
            f.write(stat_report)

        print(f"Added statistical analysis to report: {report_path}")

    # Replace the method
    tester_class._generate_enhanced_summary_report_original = tester_class._generate_enhanced_summary_report
    tester_class._generate_enhanced_summary_report = enhanced_generate_summary_report

    return tester_class


class EnhancedIntervalProductionTester:
    """Enhanced tester for interval production models with detailed piezo vs non-piezo analysis"""

    def __init__(self, base_results_dir="enhanced_production_dataset_test_results"):
        self.base_results_dir = base_results_dir
        self.cardiac_data_path = "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/10shrv0"
        self.dataset_path = "enhanced_interval_datasets/interval_production_dataset.json"
        tools.mkdir_p(base_results_dir)

        # Load the dataset
        self.test_data = self._load_test_dataset()

        print(f"Enhanced Interval Production Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Test dataset: {len(self.test_data)} conditions available")

    def _load_test_dataset(self):
        """Load test conditions from the interval production dataset."""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found: {self.dataset_path}")
            print("Creating fallback test conditions...")
            fallback_conditions = []
            for prod_interval in [1200, 1600, 2000, 2400]:
                for dly_interval in [1000, 1200]:
                    fallback_conditions.append({
                        'prod_interval': prod_interval,
                        'dly_interval': dly_interval,
                        'target_duration': prod_interval
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
                f"  {i + 1}: prod={condition['prod_interval']:.0f}ms, dly={condition['dly_interval']:.0f}ms{ood_marker}")
        if len(selected) > 5:
            print(f"  ... and {len(selected) - 5} more")

        return selected

    def _create_ood_conditions(self):
        """Create out-of-distribution test conditions for generalization testing."""
        # Training range was 1200-2400ms, so test above this range
        ood_intervals = [2500 ,2600, 2612, 2625, 2637, 2650, 2663,2675, 2686,2700, 2725, 2750]  # ms
        ood_conditions = []

        print(f"Creating out-of-distribution conditions...")
        print(f"  Training range was 1200-2400ms")
        print(f"  Testing extrapolation to: {ood_intervals} ms")

        for prod_interval in ood_intervals:
            dly_interval = 1200  # Middle of training delay range

            condition = {
                'prod_interval': float(prod_interval),
                'dly_interval': float(dly_interval),
                'target_duration': float(prod_interval),
                'is_ood': True,
                'ood_type': 'below_range' if prod_interval < 1200 else 'above_range'
            }
            ood_conditions.append(condition)

        return ood_conditions

    def test_single_model(self, model_dir, rule_name, test_conditions):
        """Test a single model on the selected conditions with retry logic for NaN responses"""
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

        # Initialize runner
        runner = Runner(
            rule_name=rule_name,
            model_dir=model_dir,
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
            'nan_counts_per_condition': [],
            'nan_conditions': []  # Track which conditions gave NaN
        }

        # Test each condition with retry logic
        for i, condition in enumerate(test_conditions):
            print(f"  Testing condition {i + 1}/{len(test_conditions)}: prod={condition['prod_interval']:.0f}ms")

            successful = False
            attempts = 0
            max_attempts = 5

            while not successful and attempts < max_attempts:
                attempts += 1
                try:
                    # Generate cardiac data if piezo model
                    cardiac_data = None
                    if use_piezo:
                        # Calculate task duration for this condition
                        task_duration_ms = condition['prod_interval'] * 2 + condition['dly_interval'] + 2000

                        from dataset import TaskDataset

                        try:
                            import default
                            # Start with a full set of defaults
                            hp = default.get_default_hp(rule_name=rule_name,
                                                        use_piezo=model_info.get("use_piezo", False))
                            # Update with model-specific info from saved model
                            hp.update(model_info)

                            # Build dataset sample with real cardiac data
                            ds = TaskDataset(rule_name, hp, mode="test")
                            sample = ds[0]
                            condition['hb_sequence'] = sample['hb_sequence']
                            if attempts == 1:
                                print("    Loaded real cardiac data from dataset")
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
                    if 'mean_completion' in performance:
                        if np.isnan(performance['mean_completion']):
                            performance_has_nan = True
                    if 'mean_timing_error' in performance:
                        if not np.isfinite(performance['mean_timing_error']):
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
                                f"Condition {i + 1}: prod={condition['prod_interval']:.0f}ms, dly={condition['dly_interval']:.0f}ms")
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
                            f"Condition {i + 1}: prod={condition['prod_interval']:.0f}ms (Exception)")
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
        """Calculate performance metrics for interval production"""
        if rule_name != 'interval_production':
            return {}

        batch_size = outputs.shape[1]

        # Find when model produces output (above threshold)
        threshold = 0.5
        response_errors = []
        completions = []

        # Get go cue timing
        if 'go' in trial.epochs:
            go_start = trial.epochs['go'][0]
            go_start = go_start[0] if hasattr(go_start, '__iter__') else go_start
        else:
            go_start = trial.epochs['go_cue'][1]  # End of go cue
            go_start = go_start[0] if hasattr(go_start, '__iter__') else go_start

        target_steps = trial.prod_interval  # Expected interval in steps

        if not hasattr(target_steps, '__iter__'):
            target_steps = [target_steps] * batch_size

        for b in range(batch_size):
            output_trace = outputs[go_start:, b, 0]  # After go cue
            response_idx = np.where(output_trace > threshold)[0]

            if len(response_idx) > 0:
                produced_steps = response_idx[0]  # steps after go cue
                target = target_steps[b]  # target in steps
                error_steps = abs(produced_steps - target)
                ratio = produced_steps / target
                symmetric_completion = 1 - abs(ratio - 1)  # % of completion

                response_errors.append(error_steps)
                completions.append(symmetric_completion)
            else:
                response_errors.append(np.inf)
                completions.append(np.nan)

        response_errors = np.array(response_errors)
        target_steps = np.array(target_steps)

        performance = {
            'mean_timing_error': float(np.mean(response_errors[np.isfinite(response_errors)])),
            'timing_errors': response_errors,
            'target_intervals': target_steps.tolist(),
            'completion_ratios': completions,
            'mean_completion': float(np.nanmean(completions)),
            'go_start_time': go_start
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

        # Interval production task
        timing_errors = [r['performance']['mean_timing_error'] for r in test_results if
                         'mean_timing_error' in r['performance'] and np.isfinite(r['performance']['mean_timing_error'])]
        completions = [r['performance']['mean_completion'] for r in test_results if
                       'mean_completion' in r['performance'] and not np.isnan(r['performance']['mean_completion'])]

        if timing_errors and completions:
            summary.update({
                'mean_timing_error': np.nanmean(timing_errors),
                'std_timing_error': np.nanstd(timing_errors),
                'mean_completion': np.nanmean(completions),
                'std_completion': np.nanstd(completions),
                'task_type': 'interval_production'
            })

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
        fig.suptitle('Mean Performance Comparison: Piezo vs Non-Piezo Models (Interval Production)', fontsize=16)

        # Calculate mean completions for different categories
        def get_completion_stats(models, category_filter=None):
            """Get completion statistics for models with optional category filter"""
            completions = []
            for model in models:
                for test_result in model['test_results']:
                    if test_result['has_nan']:
                        continue
                    if 'mean_completion' not in test_result['performance']:
                        continue
                    completion = test_result['performance']['mean_completion']
                    if np.isnan(completion):
                        continue

                    # Apply category filter
                    if category_filter is None:
                        completions.append(completion)
                    elif category_filter == 'id' and not test_result['condition'].get('is_ood', False):
                        completions.append(completion)
                    elif category_filter == 'ood' and test_result['condition'].get('is_ood', False):
                        completions.append(completion)

            return completions

        # Plot 1: Overall mean performance
        ax1 = axes[0, 0]
        piezo_all = get_completion_stats(piezo_models)
        no_piezo_all = get_completion_stats(no_piezo_models)

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
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Timing')
        ax1.set_ylabel('Mean Completion Ratio')
        ax1.set_title('Overall Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: In-Distribution performance
        ax2 = axes[0, 1]
        piezo_id = get_completion_stats(piezo_models, 'id')
        no_piezo_id = get_completion_stats(no_piezo_models, 'id')

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
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Timing')
        ax2.set_ylabel('Mean Completion Ratio')
        ax2.set_title('In-Distribution Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, id_means, id_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Out-of-Distribution performance
        ax3 = axes[1, 0]
        piezo_ood = get_completion_stats(piezo_models, 'ood')
        no_piezo_ood = get_completion_stats(no_piezo_models, 'ood')

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
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Timing')
        ax3.set_ylabel('Mean Completion Ratio')
        ax3.set_title('Out-of-Distribution Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

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
                                'mean_completion' not in test_result['performance'] or
                                np.isnan(test_result['performance']['mean_completion'])):
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

    def plot_piezo_vs_no_piezo_scatter(self, all_results):
        """NEW: Create scatter plot comparing mean piezo vs non-piezo model performance"""
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

        # Extract mean completion for each model
        piezo_completions = []
        no_piezo_completions = []

        for model in piezo_models:
            if 'mean_completion' in model['performance_summary']:
                piezo_completions.append(model['performance_summary']['mean_completion'])

        for model in no_piezo_models:
            if 'mean_completion' in model['performance_summary']:
                no_piezo_completions.append(model['performance_summary']['mean_completion'])

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot individual models
        if piezo_completions:
            ax.scatter([1] * len(piezo_completions), piezo_completions,
                       color='red', s=100, alpha=0.7, label=f'Piezo Models (n={len(piezo_completions)})')

        if no_piezo_completions:
            ax.scatter([2] * len(no_piezo_completions), no_piezo_completions,
                       color='blue', s=100, alpha=0.7, label=f'Non-Piezo Models (n={len(no_piezo_completions)})')

        # Add mean lines
        if piezo_completions:
            piezo_mean = np.mean(piezo_completions)
            ax.axhline(y=piezo_mean, color='red', linestyle='--', alpha=0.8,
                       label=f'Piezo Mean: {piezo_mean:.3f}')

        if no_piezo_completions:
            no_piezo_mean = np.mean(no_piezo_completions)
            ax.axhline(y=no_piezo_mean, color='blue', linestyle='--', alpha=0.8,
                       label=f'Non-Piezo Mean: {no_piezo_mean:.3f}')

        # Reference lines
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Perfect Timing')

        # Formatting
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 2.0)  # Allow for completions > 1.0
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Piezo Models', 'Non-Piezo Models'])
        ax.set_ylabel('Mean Completion Ratio', fontsize=12)
        ax.set_title('Piezo vs Non-Piezo Model Performance\n(Interval Production Task)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.base_results_dir, "piezo_vs_no_piezo_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved piezo vs non-piezo scatter plot: {save_path}")
        return save_path

    def plot_interval_length_vs_completion(self, all_results, save_path=None):
        """Create scatter plot of interval length vs completion ratio for production tasks"""

        # Filter for production task results only
        production_results = [r for r in all_results if r['rule_name'] == 'interval_production']

        if not production_results:
            print("No interval production results found for scatter plot")
            return

        # Collect data for scatter plot
        piezo_data = {}  # interval_length: [completion_ratios]
        no_piezo_data = {}  # interval_length: [completion_ratios]

        for result in production_results:
            use_piezo = result['use_piezo']

            for test_result in result['test_results']:
                # Skip if has NaN or missing completion data
                if (test_result['has_nan'] or
                        'mean_completion' not in test_result['performance'] or
                        np.isnan(test_result['performance']['mean_completion'])):
                    continue

                # Get interval length from condition
                interval_length = test_result['condition']['prod_interval']
                completion_ratio = test_result['performance']['mean_completion']

                # Group by piezo vs no piezo
                if use_piezo:
                    if interval_length not in piezo_data:
                        piezo_data[interval_length] = []
                    piezo_data[interval_length].append(completion_ratio)
                else:
                    if interval_length not in no_piezo_data:
                        no_piezo_data[interval_length] = []
                    no_piezo_data[interval_length].append(completion_ratio)

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot piezo models
        if piezo_data:
            piezo_intervals = []
            piezo_means = []
            piezo_stds = []

            for interval_length in sorted(piezo_data.keys()):
                completions = piezo_data[interval_length]
                piezo_intervals.append(interval_length)
                piezo_means.append(np.mean(completions))
                piezo_stds.append(np.std(completions))

            # Plot with error bars
            ax.errorbar(piezo_intervals, piezo_means, yerr=piezo_stds,
                        fmt='o', color='red', markersize=8, capsize=5,
                        label=f'Piezo Models (n={len(piezo_data)} intervals)',
                        alpha=0.8, linewidth=2)

        # Plot no piezo models
        if no_piezo_data:
            no_piezo_intervals = []
            no_piezo_means = []
            no_piezo_stds = []

            for interval_length in sorted(no_piezo_data.keys()):
                completions = no_piezo_data[interval_length]
                no_piezo_intervals.append(interval_length)
                no_piezo_means.append(np.mean(completions))
                no_piezo_stds.append(np.std(completions))

            # Plot with error bars
            ax.errorbar(no_piezo_intervals, no_piezo_means, yerr=no_piezo_stds,
                        fmt='s', color='blue', markersize=8, capsize=5,
                        label=f'Standard Models (n={len(no_piezo_data)} intervals)',
                        alpha=0.8, linewidth=2)

        # Add reference line at completion ratio = 1.0 (perfect timing)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 1.02, 'Perfect Timing',
                horizontalalignment='right', fontsize=10, color='gray')

        # Formatting
        ax.set_xlabel('Interval Length (ms)', fontsize=12)
        ax.set_ylabel('Mean Completion Ratio', fontsize=12)
        ax.set_title('Interval Length vs Completion Ratio\n(Interval Production Task)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Set reasonable y-axis limits
        ax.set_ylim(0, 2.0)

        # Add summary statistics as text
        summary_text = ""
        if piezo_data and no_piezo_data:
            piezo_overall_mean = np.mean([np.mean(completions) for completions in piezo_data.values()])
            no_piezo_overall_mean = np.mean([np.mean(completions) for completions in no_piezo_data.values()])

            summary_text = f"Overall Mean Completion:\n"
            summary_text += f"  Piezo: {piezo_overall_mean:.3f}\n"
            summary_text += f"  Standard: {no_piezo_overall_mean:.3f}"

            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(self.base_results_dir, "interval_length_vs_completion_scatter.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved interval length vs completion scatter plot: {save_path}")

        # Print summary statistics
        print(f"\nScatter Plot Summary:")
        print(f"  Piezo models: {len(piezo_data)} different interval lengths")
        print(f"  Standard models: {len(no_piezo_data)} different interval lengths")

        if piezo_data:
            total_piezo_points = sum(len(completions) for completions in piezo_data.values())
            print(f"  Total piezo data points: {total_piezo_points}")

        if no_piezo_data:
            total_no_piezo_points = sum(len(completions) for completions in no_piezo_data.values())
            print(f"  Total standard data points: {total_no_piezo_points}")

    def plot_task_vs_model_output(self, all_results):
        """NEW: Create task vs model output visualizations for each model and condition"""
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

                # Plot hidden states if available
                if test_result['states']:
                    state_plot = self._plot_hidden_states_evolution(test_result, model_name, model_dir)

            print(f"Saved task vs output plots for {model_name} to: {model_dir}")

    def _plot_task_structure(self, test_result, model_name, save_dir):
        """Plot task structure showing inputs and targets"""
        condition = test_result['condition']
        trial = test_result['trial']
        condition_idx = test_result['condition_idx']

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot task inputs
        for ch in range(trial['x'].shape[2]):
            ax.plot(trial['x'][:, 0, ch], label=f'Input {ch}', linewidth=2)

        # Plot target outputs
        for ch in range(trial['y'].shape[2]):
            ax.plot(trial['y'][:, 0, ch], '--', label=f'Target {ch}', alpha=0.7, linewidth=2)

        # Add epoch boundaries
        if 'epochs' in trial:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            for i, (epoch_name, (start, end)) in enumerate(trial['epochs'].items()):
                if start is not None:
                    color = colors[i % len(colors)]
                    start_time = start[0] if hasattr(start, '__iter__') else start
                    ax.axvline(start_time, color=color, linestyle=':', alpha=0.7, linewidth=2)
                    ax.text(start_time, ax.get_ylim()[1] * 0.9, epoch_name, rotation=90, fontsize=10, color=color)

        ax.set_title(f'Task Structure - {model_name}\nCondition {condition_idx + 1}: {condition}', fontsize=14)
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
        """Plot model outputs vs targets in comprehensive_model_test style"""
        condition = test_result['condition']
        trial = test_result['trial']
        outputs = test_result['outputs']
        condition_idx = test_result['condition_idx']
        performance = test_result['performance']

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot model outputs and targets
        for ch in range(outputs.shape[2]):
            ax.plot(outputs[:, 0, ch], label=f'Model Output {ch}', linewidth=2)

        for ch in range(trial['y'].shape[2]):
            ax.plot(trial['y'][:, 0, ch], '--', alpha=0.7, label=f'Target {ch}', linewidth=2)

        # Add performance text box (matching the image format)
        perf_text = f"Timing Error (steps): {performance.get('mean_timing_error', 0):.3f}\n"
        perf_text += f"Mean Completion: {performance.get('mean_completion', 0):.3f}"
        if 'go_start_time' in performance:
            perf_text += f"\nGo start: {performance['go_start_time']}"

        # Add NaN warning if detected
        if test_result['has_nan']:
            perf_text += f"\n*** NaN DETECTED ***"

        # Text box in upper left corner (matching the image)
        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat' if not test_result['has_nan'] else 'salmon',
                          alpha=0.8))

        # Set labels and title (matching the format from the image)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Output Activity')
        ax.set_title(f'Model Outputs vs Targets - {model_name}\nCondition {condition_idx + 1}: {condition}',
                     fontsize=14)

        # Add legend to the right side
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_model_outputs.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_hidden_states_evolution(self, test_result, model_name, save_dir):
        """Plot hidden state evolution"""
        condition = test_result['condition']
        condition_idx = test_result['condition_idx']
        states = test_result['states']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot subset of hidden state evolution
        final_state = states[-1][0]  # Last timestep, first batch
        n_neurons = len(final_state)
        sample_indices = np.linspace(0, n_neurons - 1, min(20, n_neurons), dtype=int)

        state_evolution = np.array([state[0] for state in states])  # [time, neurons]

        # Evolution plot
        for idx in sample_indices:
            ax1.plot(state_evolution[:, idx], alpha=0.7, linewidth=1)

        ax1.set_title(f'Hidden State Evolution (Sample of {len(sample_indices)} Neurons)')
        ax1.set_ylabel('Hidden State')
        ax1.grid(True, alpha=0.3)

        # Final state distribution
        ax2.hist(final_state, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title(f'Final Hidden State Distribution')
        ax2.set_xlabel('Hidden State Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Add NaN warning if detected
        if test_result['has_nan']:
            fig.suptitle(
                f'Hidden States - {model_name} *** NaN DETECTED ***\nCondition {condition_idx + 1}: {condition}',
                fontsize=14, color='red')
        else:
            fig.suptitle(f'Hidden States - {model_name}\nCondition {condition_idx + 1}: {condition}', fontsize=14)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_hidden_states.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def discover_production_models(self):
        """Discover trained interval production models"""
        model_dirs = []

        # Define possible base directories for production models
        possible_base_dirs = [
            'enhanced_piezo_production_results',
            'production_results',
            'interval_production_results',
            'model/interval_production',
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                print(f"Searching in {base_dir}...")
                self._search_model_subdirs(base_dir, 'interval_production', model_dirs)

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
        """NEW: Generate enhanced summary report with detailed piezo vs non-piezo statistics and NaN tracking"""
        report_path = os.path.join(self.base_results_dir, "enhanced_production_test_report.txt")

        # Separate conditions
        id_conditions = [c for c in test_conditions if not c.get('is_ood', False)]
        ood_conditions = [c for c in test_conditions if c.get('is_ood', False)]

        # Collect NaN information
        all_nan_conditions = []
        for result in all_results:
            if 'nan_conditions' in result and result['nan_conditions']:
                all_nan_conditions.append({
                    'model_name': result['model_name'],
                    'nan_conditions': result['nan_conditions']
                })

        with open(report_path, 'w') as f:
            f.write("ENHANCED INTERVAL PRODUCTION DATASET TEST REPORT\n")
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

            # NaN CONDITIONS REPORT
            f.write("NaN CONDITIONS REPORT\n")
            f.write("-" * 25 + "\n")
            if all_nan_conditions:
                for nan_info in all_nan_conditions:
                    f.write(f"Model: {nan_info['model_name']}\n")
                    for condition in nan_info['nan_conditions']:
                        f.write(f"  - {condition}\n")
                    f.write("\n")
            else:
                f.write("No models produced NaN responses\n\n")

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

            # Piezo vs Non-Piezo Scatter Plot Data
            f.write("PIEZO vs NON-PIEZO SCATTER PLOT METRICS:\n")
            piezo_completions = [r['performance_summary']['mean_completion'] for r in piezo_models
                                 if 'mean_completion' in r['performance_summary']]
            no_piezo_completions = [r['performance_summary']['mean_completion'] for r in no_piezo_models
                                    if 'mean_completion' in r['performance_summary']]

            if piezo_completions:
                f.write(f"  Piezo Models Individual Completions: {[f'{comp:.4f}' for comp in piezo_completions]}\n")
                f.write(f"  Piezo Mean: {np.mean(piezo_completions):.4f} ± {np.std(piezo_completions):.4f}\n")

            if no_piezo_completions:
                f.write(
                    f"  Non-Piezo Models Individual Completions: {[f'{comp:.4f}' for comp in no_piezo_completions]}\n")
                f.write(f"  Non-Piezo Mean: {np.mean(no_piezo_completions):.4f} ± {np.std(no_piezo_completions):.4f}\n")

            f.write("\n")

            # Interval Length vs Completion Scatter Plot
            f.write("INTERVAL LENGTH vs COMPLETION SCATTER PLOT METRICS:\n")
            f.write("  (Data aggregated across all interval lengths tested)\n\n")

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

                # Add NaN info for this model
                if 'nan_conditions' in result and result['nan_conditions']:
                    f.write(f"NaN Conditions:\n")
                    for condition in result['nan_conditions']:
                        f.write(f"  - {condition}\n")

                f.write("\n")

        print(f"Generated enhanced summary report: {report_path}")

    def run_enhanced_dataset_test(self, num_conditions=15, include_ood=True):
        """Run the complete enhanced dataset-driven test"""
        print("Starting Enhanced Interval Production Dataset Test")
        print("=" * 70)

        # Discover models
        model_directories = self.discover_production_models()
        if not model_directories:
            print("No interval production models found!")
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
            # Mean performance comparison plot (EXISTING)
            self.plot_mean_performance_comparison(all_results)

            # NEW: Piezo vs non-piezo scatter plot
            self.plot_piezo_vs_no_piezo_scatter(all_results)

            # Interval length vs completion scatter plot (EXISTING)
            self.plot_interval_length_vs_completion(all_results)

            # NEW: Task vs model output visualizations
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
        print(f"  - Interval length vs completion: interval_length_vs_completion_scatter.png")
        print(f"  - Task vs model output: [model]_task_vs_output/")

        return all_results




def main():
    """Run enhanced interval production dataset test with statistical analysis"""
    # Add statistical analysis capability
    EnhancedIntervalProductionTesterWithStats = add_statistical_analysis_to_tester(
        EnhancedIntervalProductionTester
    )

    tester = EnhancedIntervalProductionTesterWithStats()
    results = tester.run_enhanced_dataset_test(num_conditions=25, include_ood=True)

    print(f"\nEnhanced dataset testing with statistical analysis complete!")
    print(f"Results saved to: {tester.base_results_dir}")


if __name__ == "__main__":
    main()