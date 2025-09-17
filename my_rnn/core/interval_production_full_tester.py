"""
Enhanced Interval Production Full Tester - UNIFIED VERSION
=========================================================

Tests all three types of trained interval production models with comprehensive analysis:
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import Runner
import tools
import task
import default


class ThreeWayStatisticalAnalyzer:
    """Statistical analysis module for 3-way model comparisons (Standard vs Piezo vs Insula)"""

    def __init__(self):
        self.results = {}

    def extract_performance_data(self, all_results, task_type='interval_production'):
        """Extract individual performance data points for 3-way statistical analysis"""
        standard_data = {
            'overall': [],
            'in_distribution': [],
            'out_of_distribution': [],
            'model_means': []
        }
        
        piezo_data = {
            'overall': [],
            'in_distribution': [],
            'out_of_distribution': [],
            'model_means': []
        }
        
        insula_data = {
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
            model_type = result['model_type']  # 'standard', 'piezo', or 'insula'
            
            if model_type == 'standard':
                data_dict = standard_data
            elif model_type == 'piezo':
                data_dict = piezo_data
            else:  # insula
                data_dict = insula_data

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

        return standard_data, piezo_data, insula_data

    def perform_three_way_statistical_tests(self, standard_data, piezo_data, insula_data, task_type='interval_production'):
        """Perform comprehensive 3-way statistical testing"""
        results = {}

        # Performance metric name for reporting
        if task_type == 'interval_production':
            metric_name = 'Completion Ratio'
        else:
            metric_name = 'Accuracy'

        categories = ['overall', 'in_distribution', 'out_of_distribution', 'model_means']

        for category in categories:
            standard_vals = np.array(standard_data[category])
            piezo_vals = np.array(piezo_data[category])
            insula_vals = np.array(insula_data[category])

            if len(standard_vals) == 0 or len(piezo_vals) == 0 or len(insula_vals) == 0:
                results[category] = {'error': 'Insufficient data for one or more groups'}
                continue

            # Descriptive statistics
            desc_stats = {
                'standard_n': len(standard_vals),
                'standard_mean': np.mean(standard_vals),
                'standard_std': np.std(standard_vals, ddof=1),
                'standard_median': np.median(standard_vals),
                'piezo_n': len(piezo_vals),
                'piezo_mean': np.mean(piezo_vals),
                'piezo_std': np.std(piezo_vals, ddof=1),
                'piezo_median': np.median(piezo_vals),
                'insula_n': len(insula_vals),
                'insula_mean': np.mean(insula_vals),
                'insula_std': np.std(insula_vals, ddof=1),
                'insula_median': np.median(insula_vals),
                'overall_mean': np.mean(np.concatenate([standard_vals, piezo_vals, insula_vals])),
                'overall_std': np.std(np.concatenate([standard_vals, piezo_vals, insula_vals]), ddof=1)
            }

            # Normality tests (Shapiro-Wilk for each group)
            _, standard_normality_p = stats.shapiro(standard_vals) if len(standard_vals) > 3 else (None, None)
            _, piezo_normality_p = stats.shapiro(piezo_vals) if len(piezo_vals) > 3 else (None, None)
            _, insula_normality_p = stats.shapiro(insula_vals) if len(insula_vals) > 3 else (None, None)

            # Homogeneity of variance test (Levene's test)
            _, variance_equality_p = stats.levene(standard_vals, piezo_vals, insula_vals)

            # Check ANOVA assumptions
            assume_normal = (
                (standard_normality_p is None or standard_normality_p > 0.05) and
                (piezo_normality_p is None or piezo_normality_p > 0.05) and
                (insula_normality_p is None or insula_normality_p > 0.05)
            )
            assume_equal_var = variance_equality_p > 0.05

            # Main statistical test
            if assume_normal and assume_equal_var:
                # ANOVA
                f_stat, anova_p = stats.f_oneway(standard_vals, piezo_vals, insula_vals)
                
                # Effect size (eta-squared)
                ss_between = len(standard_vals) * (desc_stats['standard_mean'] - desc_stats['overall_mean'])**2 + \
                           len(piezo_vals) * (desc_stats['piezo_mean'] - desc_stats['overall_mean'])**2 + \
                           len(insula_vals) * (desc_stats['insula_mean'] - desc_stats['overall_mean'])**2
                ss_total = np.sum((np.concatenate([standard_vals, piezo_vals, insula_vals]) - desc_stats['overall_mean'])**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                parametric_test = {
                    'test_name': 'One-way ANOVA',
                    'f_statistic': f_stat,
                    'p_value': anova_p,
                    'eta_squared': eta_squared,
                    'degrees_of_freedom': (2, len(standard_vals) + len(piezo_vals) + len(insula_vals) - 3)
                }
            else:
                parametric_test = None

            # Non-parametric test (Kruskal-Wallis)
            h_stat, kw_p = stats.kruskal(standard_vals, piezo_vals, insula_vals)
            
            # Effect size for Kruskal-Wallis (epsilon-squared)
            n_total = len(standard_vals) + len(piezo_vals) + len(insula_vals)
            epsilon_squared = (h_stat - 2) / (n_total - 3) if n_total > 3 else 0
            
            nonparametric_test = {
                'test_name': 'Kruskal-Wallis',
                'h_statistic': h_stat,
                'p_value': kw_p,
                'epsilon_squared': epsilon_squared,
                'degrees_of_freedom': 2
            }

            # Pairwise comparisons
            pairwise_results = self._perform_pairwise_comparisons(
                standard_vals, piezo_vals, insula_vals, assume_normal and assume_equal_var
            )

            results[category] = {
                'descriptives': desc_stats,
                'assumptions': {
                    'standard_normality_p': standard_normality_p,
                    'piezo_normality_p': piezo_normality_p,
                    'insula_normality_p': insula_normality_p,
                    'variance_equality_p': variance_equality_p,
                    'assume_normal': assume_normal,
                    'assume_equal_var': assume_equal_var,
                    'use_parametric': assume_normal and assume_equal_var
                },
                'parametric': parametric_test,
                'nonparametric': nonparametric_test,
                'pairwise': pairwise_results
            }

        return results

    def _perform_pairwise_comparisons(self, standard_vals, piezo_vals, insula_vals, use_parametric):
        """Perform pairwise comparisons between the three groups"""
        comparisons = [
            ('Standard', 'Piezo', standard_vals, piezo_vals),
            ('Standard', 'Insula', standard_vals, insula_vals),
            ('Piezo', 'Insula', piezo_vals, insula_vals)
        ]
        
        pairwise_results = {}
        p_values = []
        comparison_names = []
        
        for group1_name, group2_name, group1_vals, group2_vals in comparisons:
            comparison_key = f"{group1_name}_vs_{group2_name}"
            comparison_names.append(comparison_key)
            
            if use_parametric:
                # Independent t-test
                t_stat, t_p = stats.ttest_ind(group1_vals, group2_vals, equal_var=True)
                
                # Cohen's d
                pooled_std = np.sqrt(((len(group1_vals) - 1) * np.std(group1_vals, ddof=1)**2 +
                                     (len(group2_vals) - 1) * np.std(group2_vals, ddof=1)**2) /
                                    (len(group1_vals) + len(group2_vals) - 2))
                cohens_d = (np.mean(group1_vals) - np.mean(group2_vals)) / pooled_std
                
                pairwise_results[comparison_key] = {
                    'test_type': 't-test',
                    'statistic': t_stat,
                    'p_value': t_p,
                    'effect_size': cohens_d,
                    'effect_size_name': "Cohen's d"
                }
                p_values.append(t_p)
            else:
                # Mann-Whitney U test
                u_stat, u_p = stats.mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
                
                # Effect size r = Z/sqrt(N)
                n1, n2 = len(group1_vals), len(group2_vals)
                mean_u = n1 * n2 / 2
                std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                z_score = (u_stat - mean_u) / std_u
                r_effect = abs(z_score) / np.sqrt(n1 + n2)
                
                pairwise_results[comparison_key] = {
                    'test_type': 'Mann-Whitney U',
                    'statistic': u_stat,
                    'p_value': u_p,
                    'effect_size': r_effect,
                    'effect_size_name': 'r'
                }
                p_values.append(u_p)
        
        # Apply Bonferroni correction
        corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        
        for i, comparison_key in enumerate(comparison_names):
            pairwise_results[comparison_key]['corrected_p_value'] = corrected_p_values[i]
        
        return pairwise_results

    def interpret_effect_size(self, effect_size, effect_type):
        """Interpret effect sizes using standard conventions"""
        abs_effect = abs(effect_size)
        
        if effect_type == "Cohen's d":
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        elif effect_type == "eta_squared" or effect_type == "epsilon_squared":
            if abs_effect < 0.01:
                return "small"
            elif abs_effect < 0.06:
                return "medium"
            else:
                return "large"
        elif effect_type == "r":
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        
        return "unknown"

    def generate_three_way_statistical_report(self, results, task_type='interval_production'):
        """Generate comprehensive 3-way statistical analysis report"""
        if task_type == 'interval_production':
            metric_name = 'Completion Ratio'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be significant differences in mean completion between the three model types (overall)
2. There will be significant differences in mean completion between the three model types (in-distribution)
3. There will be significant differences in mean completion between the three model types (out-of-distribution)
4. Pairwise comparisons: Standard vs Piezo, Standard vs Insula, Piezo vs Insula
"""
        else:
            metric_name = 'Accuracy'
            hypothesis_text = """
RESEARCH HYPOTHESES:
1. There will be significant differences in accuracy between the three model types (overall)
2. There will be significant differences in accuracy between the three model types (in-distribution)  
3. There will be significant differences in accuracy between the three model types (out-of-distribution)
4. Pairwise comparisons: Standard vs Piezo, Standard vs Insula, Piezo vs Insula
"""

        # Dynamic title based on task type
        if task_type == 'interval_production':
            task_title = "INTERVAL PRODUCTION MODELS"
        elif task_type == 'interval_comparison':
            task_title = "INTERVAL COMPARISON MODELS"
        elif task_type == 'time_bisection':
            task_title = "TIME BISECTION MODELS"
        else:
            task_title = "MODELS"

        report = f"""
COMPREHENSIVE 3-WAY STATISTICAL ANALYSIS - {task_title}
====================================================================
{hypothesis_text}
STATISTICAL APPROACH:
- One-way ANOVA for normally distributed data with equal variances
- Kruskal-Wallis test for non-normal data or unequal variances
- Pairwise comparisons: t-tests (parametric) or Mann-Whitney U (non-parametric)
- Bonferroni correction for multiple pairwise comparisons
- Effect size calculations (η², ε², Cohen's d, r)

MODEL TYPES:
- Standard Models: No piezo, no insula (baseline)
- Piezo Models: Piezo enabled, no insula (cardiac signal input)
- Insula Models: Insula enabled, no piezo (insula region simulation)

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
            report += f"  Sample Sizes: Standard n={desc['standard_n']}, Piezo n={desc['piezo_n']}, Insula n={desc['insula_n']}\n"
            report += f"  Means: Standard={desc['standard_mean']:.4f} (SD={desc['standard_std']:.4f}), " \
                      f"Piezo={desc['piezo_mean']:.4f} (SD={desc['piezo_std']:.4f}), " \
                      f"Insula={desc['insula_mean']:.4f} (SD={desc['insula_std']:.4f})\n"

            # Assumption tests
            report += f"\n  Assumption Tests:\n"
            if assumptions['standard_normality_p'] is not None:
                report += f"    Normality (Shapiro-Wilk): Standard p={assumptions['standard_normality_p']:.4f}, " \
                          f"Piezo p={assumptions['piezo_normality_p']:.4f}, " \
                          f"Insula p={assumptions['insula_normality_p']:.4f}\n"
            report += f"    Equal Variances (Levene): p={assumptions['variance_equality_p']:.4f}\n"
            report += f"    Using {'Parametric' if assumptions['use_parametric'] else 'Non-parametric'} tests\n"

            # Main statistical test
            if assumptions['use_parametric'] and result['parametric']:
                param = result['parametric']
                report += f"\n  Main Test ({param['test_name']}):\n"
                report += f"    F({param['degrees_of_freedom'][0]},{param['degrees_of_freedom'][1]}) = {param['f_statistic']:.4f}, p = {param['p_value']:.4f}\n"
                report += f"    Effect size η² = {param['eta_squared']:.4f} ({self.interpret_effect_size(param['eta_squared'], 'eta_squared')} effect)\n"
            else:
                nonparam = result['nonparametric']
                report += f"\n  Main Test ({nonparam['test_name']}):\n"
                report += f"    H({nonparam['degrees_of_freedom']}) = {nonparam['h_statistic']:.4f}, p = {nonparam['p_value']:.4f}\n"
                report += f"    Effect size ε² = {nonparam['epsilon_squared']:.4f} ({self.interpret_effect_size(nonparam['epsilon_squared'], 'epsilon_squared')} effect)\n"

            # Pairwise comparisons
            report += f"\n  Pairwise Comparisons:\n"
            for comparison_key, pairwise_result in result['pairwise'].items():
                group1, group2 = comparison_key.replace('_vs_', ' vs ').split(' vs ')
                report += f"    {group1} vs {group2} ({pairwise_result['test_type']}):\n"
                report += f"      Statistic = {pairwise_result['statistic']:.4f}, p = {pairwise_result['p_value']:.4f}"
                report += f" (corrected p = {pairwise_result['corrected_p_value']:.4f})\n"
                report += f"      Effect size {pairwise_result['effect_size_name']} = {pairwise_result['effect_size']:.4f} " \
                          f"({self.interpret_effect_size(pairwise_result['effect_size'], pairwise_result['effect_size_name'])} effect)\n"

            # Overall interpretation
            main_test = result['parametric'] if assumptions['use_parametric'] and result['parametric'] else result['nonparametric']
            main_p = main_test['p_value']
            
            if main_p < 0.001:
                sig_text = "highly significant (p < 0.001)"
            elif main_p < 0.01:
                sig_text = "very significant (p < 0.01)"
            elif main_p < 0.05:
                sig_text = "significant (p < 0.05)"
            else:
                sig_text = "not significant (p ≥ 0.05)"

            report += f"\n  INTERPRETATION: The overall difference between model types is {sig_text}.\n"

        return report


class EnhancedIntervalProductionFullTester:
    """Unified tester for all three types of interval production models"""

    def __init__(self, base_results_dir="interval_production_full_results"):
        self.base_results_dir = base_results_dir
        self.dataset_path = "enhanced_interval_datasets/interval_production_dataset.json"
        
        # Load best cardiac libraries for both piezo and insula models
        self.best_piezo_library = self._load_best_piezo_library()
        self.best_insula_library = self._load_best_insula_library()
        
        tools.mkdir_p(base_results_dir)

        # Load the dataset
        self.test_data = self._load_test_dataset()

        print(f"Enhanced Interval Production Full Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Test dataset: {len(self.test_data)} conditions available")
        print(f"Best piezo library: {self.best_piezo_library}")
        print(f"Best insula library: {self.best_insula_library}")

    def _load_best_piezo_library(self):
        """Load the best performing cardiac library for piezo models"""
        library_results_path = "interval_production_cardiac_library_results/cross_model_library_analysis.json"
        
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
        library_results_path = "interval_production_insula_library_results/cross_insula_model_library_analysis.json"
        
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
        ood_intervals = [2500, 2600, 2612, 2625, 2637, 2650, 2663, 2675, 2686, 2700, 2725, 2750]  # ms
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
                'ood_type': 'above_range'
            }
            ood_conditions.append(condition)

        return ood_conditions

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
            print(f"  Testing condition {i + 1}/{len(test_conditions)}: prod={condition['prod_interval']:.0f}ms")

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
                        task_duration_ms = condition['prod_interval'] * 2 + condition['dly_interval'] + 2000

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
        fig.suptitle('3-Way Performance Comparison: Standard vs Piezo vs Insula Models (Interval Production)', fontsize=16)

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
        standard_all = get_completion_stats(standard_models)
        piezo_all = get_completion_stats(piezo_models)
        insula_all = get_completion_stats(insula_models)

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
        standard_id = get_completion_stats(standard_models, 'id')
        piezo_id = get_completion_stats(piezo_models, 'id')
        insula_id = get_completion_stats(insula_models, 'id')

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
        standard_ood = get_completion_stats(standard_models, 'ood')
        piezo_ood = get_completion_stats(piezo_models, 'ood')
        insula_ood = get_completion_stats(insula_models, 'ood')

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

        # Extract mean completion for each model
        standard_completions = []
        piezo_completions = []
        insula_completions = []

        for model in standard_models:
            if 'mean_completion' in model['performance_summary']:
                standard_completions.append(model['performance_summary']['mean_completion'])

        for model in piezo_models:
            if 'mean_completion' in model['performance_summary']:
                piezo_completions.append(model['performance_summary']['mean_completion'])

        for model in insula_models:
            if 'mean_completion' in model['performance_summary']:
                insula_completions.append(model['performance_summary']['mean_completion'])

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot individual models
        if standard_completions:
            ax.scatter([1] * len(standard_completions), standard_completions,
                       color='blue', s=100, alpha=0.7, label=f'Standard Models (n={len(standard_completions)})')

        if piezo_completions:
            ax.scatter([2] * len(piezo_completions), piezo_completions,
                       color='red', s=100, alpha=0.7, label=f'Piezo Models (n={len(piezo_completions)})')

        if insula_completions:
            ax.scatter([3] * len(insula_completions), insula_completions,
                       color='green', s=100, alpha=0.7, label=f'Insula Models (n={len(insula_completions)})')

        # Add mean lines
        if standard_completions:
            standard_mean = np.mean(standard_completions)
            ax.axhline(y=standard_mean, color='blue', linestyle='--', alpha=0.8,
                       label=f'Standard Mean: {standard_mean:.3f}')

        if piezo_completions:
            piezo_mean = np.mean(piezo_completions)
            ax.axhline(y=piezo_mean, color='red', linestyle='--', alpha=0.8,
                       label=f'Piezo Mean: {piezo_mean:.3f}')

        if insula_completions:
            insula_mean = np.mean(insula_completions)
            ax.axhline(y=insula_mean, color='green', linestyle='--', alpha=0.8,
                       label=f'Insula Mean: {insula_mean:.3f}')

        # Reference lines
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Perfect Timing')

        # Formatting
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 2.0)  # Allow for completions > 1.0
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Standard Models', 'Piezo Models', 'Insula Models'])
        ax.set_ylabel('Mean Completion Ratio', fontsize=12)
        ax.set_title('3-Way Model Performance Comparison\n(Interval Production Task)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.base_results_dir, "three_way_model_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3-way model scatter plot: {save_path}")
        return save_path

    def plot_interval_length_vs_completion_three_way(self, all_results, save_path=None):
        """Create scatter plot of interval length vs completion ratio for all three model types"""

        # Filter for production task results only
        production_results = [r for r in all_results if r['rule_name'] == 'interval_production']

        if not production_results:
            print("No interval production results found for scatter plot")
            return

        # Collect data for scatter plot
        standard_data = {}  # interval_length: [completion_ratios]
        piezo_data = {}  # interval_length: [completion_ratios]
        insula_data = {}  # interval_length: [completion_ratios]

        for result in production_results:
            model_type = result['model_type']

            for test_result in result['test_results']:
                # Skip if has NaN or missing completion data
                if (test_result['has_nan'] or
                        'mean_completion' not in test_result['performance'] or
                        np.isnan(test_result['performance']['mean_completion'])):
                    continue

                # Get interval length from condition
                interval_length = test_result['condition']['prod_interval']
                completion_ratio = test_result['performance']['mean_completion']

                # Group by model type
                if model_type == 'standard':
                    if interval_length not in standard_data:
                        standard_data[interval_length] = []
                    standard_data[interval_length].append(completion_ratio)
                elif model_type == 'piezo':
                    if interval_length not in piezo_data:
                        piezo_data[interval_length] = []
                    piezo_data[interval_length].append(completion_ratio)
                else:  # insula
                    if interval_length not in insula_data:
                        insula_data[interval_length] = []
                    insula_data[interval_length].append(completion_ratio)

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot standard models
        if standard_data:
            standard_intervals = []
            standard_means = []
            standard_stds = []

            for interval_length in sorted(standard_data.keys()):
                completions = standard_data[interval_length]
                standard_intervals.append(interval_length)
                standard_means.append(np.mean(completions))
                standard_stds.append(np.std(completions))

            # Plot with error bars
            ax.errorbar(standard_intervals, standard_means, yerr=standard_stds,
                        fmt='o', color='blue', markersize=8, capsize=5,
                        label=f'Standard Models (n={len(standard_data)} intervals)',
                        alpha=0.8, linewidth=2)

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
                        fmt='s', color='red', markersize=8, capsize=5,
                        label=f'Piezo Models (n={len(piezo_data)} intervals)',
                        alpha=0.8, linewidth=2)

        # Plot insula models
        if insula_data:
            insula_intervals = []
            insula_means = []
            insula_stds = []

            for interval_length in sorted(insula_data.keys()):
                completions = insula_data[interval_length]
                insula_intervals.append(interval_length)
                insula_means.append(np.mean(completions))
                insula_stds.append(np.std(completions))

            # Plot with error bars
            ax.errorbar(insula_intervals, insula_means, yerr=insula_stds,
                        fmt='^', color='green', markersize=8, capsize=5,
                        label=f'Insula Models (n={len(insula_data)} intervals)',
                        alpha=0.8, linewidth=2)

        # Add reference line at completion ratio = 1.0 (perfect timing)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 1.02, 'Perfect Timing',
                horizontalalignment='right', fontsize=10, color='gray')

        # Formatting
        ax.set_xlabel('Interval Length (ms)', fontsize=12)
        ax.set_ylabel('Mean Completion Ratio', fontsize=12)
        ax.set_title('Interval Length vs Completion Ratio - 3-Way Comparison\n(Interval Production Task)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Set reasonable y-axis limits
        ax.set_ylim(0, 2.0)

        # Add summary statistics as text
        summary_text = ""
        if standard_data and piezo_data and insula_data:
            standard_overall_mean = np.mean([np.mean(completions) for completions in standard_data.values()])
            piezo_overall_mean = np.mean([np.mean(completions) for completions in piezo_data.values()])
            insula_overall_mean = np.mean([np.mean(completions) for completions in insula_data.values()])

            summary_text = f"Overall Mean Completion:\n"
            summary_text += f"  Standard: {standard_overall_mean:.3f}\n"
            summary_text += f"  Piezo: {piezo_overall_mean:.3f}\n"
            summary_text += f"  Insula: {insula_overall_mean:.3f}"

            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(self.base_results_dir, "interval_length_vs_completion_three_way_scatter.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3-way interval length vs completion scatter plot: {save_path}")

        # Print summary statistics
        print(f"\n3-Way Scatter Plot Summary:")
        print(f"  Standard models: {len(standard_data)} different interval lengths")
        print(f"  Piezo models: {len(piezo_data)} different interval lengths")
        print(f"  Insula models: {len(insula_data)} different interval lengths")

        if standard_data:
            total_standard_points = sum(len(completions) for completions in standard_data.values())
            print(f"  Total standard data points: {total_standard_points}")

        if piezo_data:
            total_piezo_points = sum(len(completions) for completions in piezo_data.values())
            print(f"  Total piezo data points: {total_piezo_points}")

        if insula_data:
            total_insula_points = sum(len(completions) for completions in insula_data.values())
            print(f"  Total insula data points: {total_insula_points}")

    def discover_all_production_models(self):
        """Discover all trained interval production models (standard, piezo, and insula)"""
        model_dirs = []

        # Define possible base directories for all model types
        possible_base_dirs = [
            'enhanced_piezo_production_results',  # Piezo models
            'enhanced_insula_production_results',  # Insula models  
            'production_results',  # Mixed
            'interval_production_results',  # Mixed
            'model/interval_production',  # Mixed
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                print(f"Searching in {base_dir}...")
                self._search_model_subdirs(base_dir, 'interval_production', model_dirs)

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

    def _find_best_model_checkpoint(self, base_model_dir, use_best_timing_error=True):
        """Find the best model checkpoint in a run directory based on timing error"""
        # Priority order: finalResult > best timing error checkpoint > highest numbered checkpoint > main directory
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

        if numbered_checkpoints and use_best_timing_error:
            # Try to find the best checkpoint based on timing error from log.json
            best_checkpoint = self._find_best_timing_error_checkpoint(base_model_dir, numbered_checkpoints)
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

    def _find_best_timing_error_checkpoint(self, base_model_dir, numbered_checkpoints):
        """Find checkpoint with lowest timing error based on individual checkpoint log.json files"""
        try:
            min_timing_error = float('inf')
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
                    
                    # Look for timing error metric (mean_rel_action_time)
                    if 'mean_rel_action_time' in checkpoint_log:
                        timing_error = checkpoint_log['mean_rel_action_time']
                        
                        if np.isfinite(timing_error) and timing_error < min_timing_error:
                            min_timing_error = timing_error
                            best_checkpoint_path = checkpoint_path
                            best_checkpoint_num = checkpoint_num
                            
                except Exception as e:
                    print(f"    Warning: Could not read {checkpoint_log_path}: {e}")
                    continue
            
            if best_checkpoint_path:
                print(f"  Selected best timing error checkpoint: {best_checkpoint_num} (timing error: {min_timing_error:.6f})")
                return best_checkpoint_path
            else:
                print(f"  Warning: No timing error data found in checkpoint logs")
                return None
            
        except Exception as e:
            print(f"  Warning: Could not determine best checkpoint from timing error: {e}")
            return None

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)

    def _generate_enhanced_summary_report(self, all_results, test_conditions):
        """Generate enhanced summary report with 3-way model statistics and NaN tracking"""
        report_path = os.path.join(self.base_results_dir, "enhanced_full_production_test_report.txt")

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
            f.write("ENHANCED INTERVAL PRODUCTION FULL DATASET TEST REPORT\n")
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
        print("Starting Enhanced Interval Production Full Dataset Test")
        print("=" * 80)

        # Discover all models
        model_directories = self.discover_all_production_models()
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
            # 3-way performance comparison plot
            self.plot_three_way_performance_comparison(all_results)

            # 3-way model scatter plot
            self.plot_three_way_scatter(all_results)

            # 3-way interval length vs completion scatter plot
            self.plot_interval_length_vs_completion_three_way(all_results)

            # Generate statistical analysis
            analyzer = ThreeWayStatisticalAnalyzer()
            standard_data, piezo_data, insula_data = analyzer.extract_performance_data(all_results, 'interval_production')
            statistical_results = analyzer.perform_three_way_statistical_tests(standard_data, piezo_data, insula_data, 'interval_production')
            
            # Generate and save statistical report
            stat_report = analyzer.generate_three_way_statistical_report(statistical_results, 'interval_production')
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
        print(f"  - 3-way interval length vs completion: interval_length_vs_completion_three_way_scatter.png")
        print(f"Generated reports:")
        print(f"  - 3-way statistical analysis: three_way_statistical_analysis.txt")
        print(f"  - Enhanced summary report: enhanced_full_production_test_report.txt")

        return all_results


def main():
    """Run enhanced interval production full dataset test with 3-way statistical analysis"""
    tester = EnhancedIntervalProductionFullTester()
    results = tester.run_enhanced_full_dataset_test(num_conditions=25, include_ood=True)

    print(f"\nEnhanced full dataset testing with 3-way statistical analysis complete!")
    print(f"Results saved to: {tester.base_results_dir}")


if __name__ == "__main__":
    main()
