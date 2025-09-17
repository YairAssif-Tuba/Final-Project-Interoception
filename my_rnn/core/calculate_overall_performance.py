#!/usr/bin/env python3
"""
Overall Performance Estimation Script

This script calculates overall performance metrics across all tasks
(interval_comparison, interval_production, time_bisection) regardless of task type.

It aggregates results from the *_full_results folders and provides:
1. Overall performance by model type (Standard, Piezo, Insula)
2. Cross-task performance comparison
3. Summary statistics and rankings
4. Detailed performance breakdown

Author: Generated for interoception modeling project
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


def parse_report_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a report text file to extract performance metrics.
    
    Args:
        file_path: Path to the report file
        
    Returns:
        Dictionary containing parsed performance data
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {
        'task_name': '',
        'overall_performance': {},
        'in_distribution_performance': {},
        'out_distribution_performance': {},
        'individual_models': {}
    }
    
    # Extract task name from file path
    if 'comparison' in file_path:
        results['task_name'] = 'interval_comparison'
    elif 'production' in file_path:
        results['task_name'] = 'interval_production'
    elif 'bisection' in file_path:
        results['task_name'] = 'time_bisection'
    
    # Parse overall performance section
    overall_match = re.search(
        r'OVERALL PERFORMANCE.*?:\n(.*?)(?=IN-DISTRIBUTION|$)', 
        content, re.DOTALL
    )
    if overall_match:
        perf_text = overall_match.group(1)
        for line in perf_text.strip().split('\n'):
            if 'Standard Models:' in line:
                match = re.search(r'Standard Models:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match:
                    results['overall_performance']['standard'] = {
                        'mean': float(match.group(1)),
                        'std': float(match.group(2))
                    }
            elif 'Piezo Models:' in line:
                match = re.search(r'Piezo Models:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match:
                    results['overall_performance']['piezo'] = {
                        'mean': float(match.group(1)),
                        'std': float(match.group(2))
                    }
            elif 'Insula Models:' in line:
                match = re.search(r'Insula Models:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match:
                    results['overall_performance']['insula'] = {
                        'mean': float(match.group(1)),
                        'std': float(match.group(2))
                    }
    
    # Parse individual model results
    individual_section = re.search(
        r'INDIVIDUAL MODEL RESULTS BY TYPE\n-+\n(.*?)$', 
        content, re.DOTALL
    )
    if individual_section:
        models_text = individual_section.group(1)
        
        # Split by model sections
        model_sections = re.split(r'\n\n(?=STANDARD MODELS:|PIEZO MODELS:|INSULA MODELS:)', models_text)
        
        for section in model_sections:
            if not section.strip():
                continue
                
            # Determine model type
            model_type = None
            if 'STANDARD MODELS:' in section:
                model_type = 'standard'
            elif 'PIEZO MODELS:' in section:
                model_type = 'piezo'
            elif 'INSULA MODELS:' in section:
                model_type = 'insula'
            
            if not model_type:
                continue
                
            # Parse individual models in this section
            model_blocks = re.split(r'\n\nModel: ', section)
            
            for block in model_blocks:
                if not block.strip() or 'MODELS:' in block:
                    continue
                    
                # Extract model name
                model_match = re.search(r'^([^\n]+)', block)
                if not model_match:
                    continue
                    
                model_name = model_match.group(1).strip()
                if model_name.startswith('Model: '):
                    model_name = model_name[7:]  # Remove "Model: " prefix
                
                # Extract performance metrics
                model_data = {'type': model_type}
                
                # Extract accuracy (for comparison and bisection tasks)
                acc_match = re.search(r'accuracy:\s*([\d.]+)', block)
                if acc_match:
                    model_data['accuracy'] = float(acc_match.group(1))
                
                # Extract mean_cost
                cost_match = re.search(r'mean_cost:\s*([\d.]+)', block)
                if cost_match:
                    model_data['mean_cost'] = float(cost_match.group(1))
                
                # Extract timing error (for production task)
                timing_match = re.search(r'mean_timing_error:\s*([\d.]+)', block)
                if timing_match:
                    model_data['mean_timing_error'] = float(timing_match.group(1))
                
                # Extract completion rate (for production task)
                completion_match = re.search(r'mean_completion:\s*([\d.]+)', block)
                if completion_match:
                    model_data['mean_completion'] = float(completion_match.group(1))
                
                # Extract NaN rate
                nan_match = re.search(r'nan_rate:\s*([\d.]+)', block)
                if nan_match:
                    model_data['nan_rate'] = float(nan_match.group(1))
                
                results['individual_models'][model_name] = model_data
    
    return results


def load_all_results(base_dir: str) -> Dict[str, Any]:
    """
    Load results from all *_full_results directories.
    
    Args:
        base_dir: Base directory containing the results folders
        
    Returns:
        Dictionary containing all parsed results
    """
    results_dirs = [
        'interval_comparison_full_results',
        'interval_production_full_results', 
        'time_bisection_full_results'
    ]
    
    all_results = {}
    
    for results_dir in results_dirs:
        dir_path = os.path.join(base_dir, results_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} not found")
            continue
            
        # Find the report file
        report_files = [f for f in os.listdir(dir_path) if f.endswith('_test_report.txt')]
        if not report_files:
            print(f"Warning: No report file found in {dir_path}")
            continue
            
        report_file = os.path.join(dir_path, report_files[0])
        task_results = parse_report_file(report_file)
        
        task_name = task_results['task_name']
        all_results[task_name] = task_results
        
        print(f"Loaded results for {task_name}: {len(task_results['individual_models'])} models")
    
    return all_results


class CrossTaskStatisticalAnalyzer:
    """Statistical analysis module for cross-task performance comparisons"""

    def __init__(self):
        self.results = {}

    def extract_cross_task_performance_data(self, all_results: Dict[str, Any]):
        """Extract performance data organized by model type across all tasks"""
        model_type_data = {
            'standard': {'accuracy': [], 'cost': [], 'completion': [], 'timing_error': [], 'tasks': []},
            'piezo': {'accuracy': [], 'cost': [], 'completion': [], 'timing_error': [], 'tasks': []},
            'insula': {'accuracy': [], 'cost': [], 'completion': [], 'timing_error': [], 'tasks': []}
        }
        
        for task_name, task_data in all_results.items():
            for model_name, model_data in task_data['individual_models'].items():
                model_type = model_data['type']
                
                if model_type in model_type_data:
                    # Add accuracy scores (for comparison and bisection tasks)
                    if 'accuracy' in model_data:
                        model_type_data[model_type]['accuracy'].append(model_data['accuracy'])
                        model_type_data[model_type]['tasks'].append(f"{task_name}_{model_name}")
                    
                    # Add cost scores (all tasks)
                    if 'mean_cost' in model_data:
                        model_type_data[model_type]['cost'].append(model_data['mean_cost'])
                    
                    # Add completion scores (production task)
                    if 'mean_completion' in model_data:
                        model_type_data[model_type]['completion'].append(model_data['mean_completion'])
                    
                    # Add timing error (production task)
                    if 'mean_timing_error' in model_data:
                        model_type_data[model_type]['timing_error'].append(model_data['mean_timing_error'])
        
        return model_type_data

    def perform_cross_task_statistical_tests(self, model_type_data: Dict[str, Dict[str, List]]):
        """Perform comprehensive statistical testing across model types for each metric"""
        results = {}
        
        metrics = ['accuracy', 'cost', 'completion', 'timing_error']
        metric_names = {
            'accuracy': 'Accuracy',
            'cost': 'Mean Cost',
            'completion': 'Completion Ratio',
            'timing_error': 'Timing Error'
        }
        
        for metric in metrics:
            # Extract data for each model type
            standard_vals = np.array(model_type_data['standard'][metric])
            piezo_vals = np.array(model_type_data['piezo'][metric])
            insula_vals = np.array(model_type_data['insula'][metric])
            
            # Skip if any group is empty
            if len(standard_vals) == 0 or len(piezo_vals) == 0 or len(insula_vals) == 0:
                results[metric] = {'error': 'Insufficient data for one or more groups'}
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
            
            results[metric] = {
                'metric_name': metric_names[metric],
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
                cohens_d = (np.mean(group1_vals) - np.mean(group2_vals)) / pooled_std if pooled_std > 0 else 0
                
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
                z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0
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
        if p_values:
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

    def generate_cross_task_statistical_report(self, statistical_results: Dict[str, Any]) -> str:
        """Generate comprehensive cross-task statistical analysis report"""
        
        report = """
CROSS-TASK STATISTICAL ANALYSIS
===============================

RESEARCH HYPOTHESES:
1. There will be significant differences in performance between model types across all tasks
2. Pairwise comparisons: Standard vs Piezo, Standard vs Insula, Piezo vs Insula
3. Effect sizes will indicate practical significance of differences

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
        
        for metric, result in statistical_results.items():
            if 'error' in result:
                report += f"\n{result.get('metric_name', metric.upper())} ANALYSIS:\n"
                report += f"  ERROR: {result['error']}\n"
                continue
            
            desc = result['descriptives']
            assumptions = result['assumptions']
            
            report += f"\n{result['metric_name'].upper()} ANALYSIS:\n"
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
            
            report += f"\n  INTERPRETATION: The overall difference between model types for {result['metric_name'].lower()} is {sig_text}.\n"
        
        return report


def plot_cross_task_performance_comparison(all_results: Dict[str, Any], overall_results: Dict[str, Any], 
                                         output_dir: str = ".") -> str:
    """Create comprehensive cross-task performance comparison plots"""
    print("Creating cross-task performance comparison plots...")
    
    # Extract performance data by model type and task
    tasks = list(all_results.keys())
    model_types = ['standard', 'piezo', 'insula']
    model_type_colors = {'standard': 'blue', 'piezo': 'red', 'insula': 'green'}
    model_type_labels = {'standard': 'Standard Models', 'piezo': 'Piezo Models', 'insula': 'Insula Models'}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Task Performance Comparison: Standard vs Piezo vs Insula Models', fontsize=16)
    
    # Plot 1: Accuracy comparison (for comparison and bisection tasks)
    ax1 = axes[0, 0]
    accuracy_data = {}
    for model_type in model_types:
        accuracy_data[model_type] = []
        
    for task_name, task_data in all_results.items():
        for model_name, model_data in task_data['individual_models'].items():
            if 'accuracy' in model_data:
                model_type = model_data['type']
                if model_type in accuracy_data:
                    accuracy_data[model_type].append(model_data['accuracy'])
    
    # Create box plots for accuracy
    accuracy_plot_data = []
    accuracy_labels = []
    accuracy_colors = []
    
    for model_type in model_types:
        if accuracy_data[model_type]:
            accuracy_plot_data.append(accuracy_data[model_type])
            accuracy_labels.append(model_type_labels[model_type])
            accuracy_colors.append(model_type_colors[model_type])
    
    if accuracy_plot_data:
        bp1 = ax1.boxplot(accuracy_plot_data, labels=accuracy_labels, patch_artist=True, 
                         showmeans=True, meanline=True)
        for patch, color in zip(bp1['boxes'], accuracy_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Across All Tasks')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add mean values as text
        for i, model_type in enumerate([mt for mt in model_types if accuracy_data[mt]]):
            mean_val = np.mean(accuracy_data[model_type])
            ax1.text(i+1, mean_val + 0.05, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cost comparison (all tasks)
    ax2 = axes[0, 1]
    cost_data = {}
    for model_type in model_types:
        cost_data[model_type] = []
        
    for task_name, task_data in all_results.items():
        for model_name, model_data in task_data['individual_models'].items():
            if 'mean_cost' in model_data:
                model_type = model_data['type']
                if model_type in cost_data:
                    cost_data[model_type].append(model_data['mean_cost'])
    
    # Create box plots for cost
    cost_plot_data = []
    cost_labels = []
    cost_colors = []
    
    for model_type in model_types:
        if cost_data[model_type]:
            cost_plot_data.append(cost_data[model_type])
            cost_labels.append(model_type_labels[model_type])
            cost_colors.append(model_type_colors[model_type])
    
    if cost_plot_data:
        bp2 = ax2.boxplot(cost_plot_data, labels=cost_labels, patch_artist=True, 
                         showmeans=True, meanline=True)
        for patch, color in zip(bp2['boxes'], cost_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Mean Cost')
        ax2.set_title('Cost Across All Tasks')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for cost
        
        # Add mean values as text
        for i, model_type in enumerate([mt for mt in model_types if cost_data[mt]]):
            mean_val = np.mean(cost_data[model_type])
            ax2.text(i+1, mean_val * 1.2, f'{mean_val:.4f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Completion ratio (production task only)
    ax3 = axes[1, 0]
    completion_data = {}
    for model_type in model_types:
        completion_data[model_type] = []
        
    for task_name, task_data in all_results.items():
        for model_name, model_data in task_data['individual_models'].items():
            if 'mean_completion' in model_data:
                model_type = model_data['type']
                if model_type in completion_data:
                    completion_data[model_type].append(model_data['mean_completion'])
    
    # Create box plots for completion
    completion_plot_data = []
    completion_labels = []
    completion_colors = []
    
    for model_type in model_types:
        if completion_data[model_type]:
            completion_plot_data.append(completion_data[model_type])
            completion_labels.append(model_type_labels[model_type])
            completion_colors.append(model_type_colors[model_type])
    
    if completion_plot_data:
        bp3 = ax3.boxplot(completion_plot_data, labels=completion_labels, patch_artist=True, 
                         showmeans=True, meanline=True)
        for patch, color in zip(bp3['boxes'], completion_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Timing')
        ax3.set_ylabel('Completion Ratio')
        ax3.set_title('Completion Ratio (Production Task)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add mean values as text
        for i, model_type in enumerate([mt for mt in model_types if completion_data[mt]]):
            mean_val = np.mean(completion_data[model_type])
            ax3.text(i+1, mean_val + 0.05, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Task-specific performance breakdown
    ax4 = axes[1, 1]
    
    # Create grouped bar chart showing mean performance by task and model type
    task_names = list(all_results.keys())
    task_display_names = [name.replace('_', ' ').title() for name in task_names]
    
    x = np.arange(len(task_names))
    width = 0.25
    
    # Calculate mean performance for each task and model type
    task_performance = {}
    for task_name in task_names:
        task_performance[task_name] = {}
        for model_type in model_types:
            performances = []
            
            for model_name, model_data in all_results[task_name]['individual_models'].items():
                if model_data['type'] == model_type:
                    # Use accuracy for comparison/bisection, completion for production
                    if 'accuracy' in model_data:
                        performances.append(model_data['accuracy'])
                    elif 'mean_completion' in model_data:
                        performances.append(model_data['mean_completion'])
            
            if performances:
                task_performance[task_name][model_type] = np.mean(performances)
            else:
                task_performance[task_name][model_type] = 0
    
    # Create bars
    for i, model_type in enumerate(model_types):
        values = [task_performance[task][model_type] for task in task_names]
        bars = ax4.bar(x + i * width, values, width, 
                      label=model_type_labels[model_type], 
                      color=model_type_colors[model_type], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_xlabel('Task')
    ax4.set_ylabel('Mean Performance')
    ax4.set_title('Task-Specific Performance by Model Type')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(task_display_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "cross_task_performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-task performance comparison plot: {plot_path}")
    return plot_path


def plot_composite_performance_scatter(overall_results: Dict[str, Any], output_dir: str = ".") -> str:
    """Create scatter plot showing composite performance scores across model types"""
    print("Creating composite performance scatter plot...")
    
    if 'individual_model_rankings' not in overall_results:
        print("No model rankings available for scatter plot")
        return ""
    
    # Extract data by model type
    model_types = ['standard', 'piezo', 'insula']
    model_type_colors = {'standard': 'blue', 'piezo': 'red', 'insula': 'green'}
    model_type_labels = {'standard': 'Standard Models', 'piezo': 'Piezo Models', 'insula': 'Insula Models'}
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for model_type in model_types:
        # Get models of this type
        type_models = [m for m in overall_results['individual_model_rankings'] if m['type'] == model_type]
        
        if not type_models:
            continue
        
        # Extract composite scores
        scores = [m['composite_score'] for m in type_models]
        
        # Create x positions (jittered for visibility)
        x_pos = [model_types.index(model_type) + 1] * len(scores)
        x_jitter = np.random.normal(0, 0.1, len(scores))  # Add jitter
        x_positions = np.array(x_pos) + x_jitter
        
        # Plot scatter
        ax.scatter(x_positions, scores, 
                  color=model_type_colors[model_type], 
                  s=100, alpha=0.7, 
                  label=f'{model_type_labels[model_type]} (n=10)')
        
        # Add mean line
        mean_score = np.mean(scores)
        ax.axhline(y=mean_score, color=model_type_colors[model_type], 
                  linestyle='--', alpha=0.8, 
                  label=f'{model_type.capitalize()} Mean: {mean_score:.3f}')
    
    # Formatting
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Standard Models', 'Piezo Models', 'Insula Models'])
    ax.set_ylabel('Composite Performance Score', fontsize=12)
    ax.set_title('Task-Weighted Composite Performance Scores\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add summary statistics with task-weighted explanation (positioned at bottom)
    summary_text = "Task-Weighted Composite Score:\n"
    summary_text += "• Primary metric: 70% weight\n"
    summary_text += "  - Accuracy (comparison/bisection)\n"
    summary_text += "  - Completion (production)\n"
    summary_text += "• Cost (inverted): 30% weight (all tasks)\n"
    summary_text += "• Production models: 2x weight\n"
    summary_text += "  (balances 60 vs 30 model representation)"
    
    ax.text(0.02, 0.35, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "composite_performance_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved composite performance scatter plot: {plot_path}")
    return plot_path


def calculate_overall_performance(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall performance metrics across all tasks.
    
    Args:
        all_results: Results from all tasks
        
    Returns:
        Dictionary containing overall performance analysis
    """
    # Collect all individual model results
    all_models = {}
    task_performance = {}
    
    for task_name, task_data in all_results.items():
        task_performance[task_name] = task_data['overall_performance']
        
        for model_name, model_data in task_data['individual_models'].items():
            if model_name not in all_models:
                all_models[model_name] = {'type': model_data['type'], 'tasks': {}}
            all_models[model_name]['tasks'][task_name] = model_data
    
    # Calculate cross-task performance metrics
    model_types = ['standard', 'piezo', 'insula']
    overall_summary = {
        'by_model_type': {},
        'by_task': task_performance,
        'cross_task_analysis': {},
        'individual_model_rankings': {}
    }
    
    for model_type in model_types:
        type_models = [name for name, data in all_models.items() if data['type'] == model_type]
        type_metrics = {
            'accuracy_scores': [],
            'cost_scores': [],
            'completion_scores': [],
            'timing_error_scores': [],
            'nan_rates': [],
            'model_count': len(type_models),
            'task_coverage': {}
        }
        
        for model_name in type_models:
            model_data = all_models[model_name]
            
            for task_name, task_metrics in model_data['tasks'].items():
                if task_name not in type_metrics['task_coverage']:
                    type_metrics['task_coverage'][task_name] = 0
                type_metrics['task_coverage'][task_name] += 1
                
                # Collect metrics
                if 'accuracy' in task_metrics:
                    type_metrics['accuracy_scores'].append(task_metrics['accuracy'])
                if 'mean_cost' in task_metrics:
                    type_metrics['cost_scores'].append(task_metrics['mean_cost'])
                if 'mean_completion' in task_metrics:
                    type_metrics['completion_scores'].append(task_metrics['mean_completion'])
                if 'mean_timing_error' in task_metrics:
                    type_metrics['timing_error_scores'].append(task_metrics['mean_timing_error'])
                if 'nan_rate' in task_metrics:
                    type_metrics['nan_rates'].append(task_metrics['nan_rate'])
        
        # Calculate summary statistics
        type_summary = {
            'model_count': type_metrics['model_count'],
            'task_coverage': type_metrics['task_coverage']
        }
        
        if type_metrics['accuracy_scores']:
            type_summary['accuracy'] = {
                'mean': np.mean(type_metrics['accuracy_scores']),
                'std': np.std(type_metrics['accuracy_scores']),
                'median': np.median(type_metrics['accuracy_scores']),
                'min': np.min(type_metrics['accuracy_scores']),
                'max': np.max(type_metrics['accuracy_scores']),
                'count': len(type_metrics['accuracy_scores'])
            }
        
        if type_metrics['cost_scores']:
            type_summary['cost'] = {
                'mean': np.mean(type_metrics['cost_scores']),
                'std': np.std(type_metrics['cost_scores']),
                'median': np.median(type_metrics['cost_scores']),
                'min': np.min(type_metrics['cost_scores']),
                'max': np.max(type_metrics['cost_scores']),
                'count': len(type_metrics['cost_scores'])
            }
        
        if type_metrics['completion_scores']:
            type_summary['completion'] = {
                'mean': np.mean(type_metrics['completion_scores']),
                'std': np.std(type_metrics['completion_scores']),
                'median': np.median(type_metrics['completion_scores']),
                'min': np.min(type_metrics['completion_scores']),
                'max': np.max(type_metrics['completion_scores']),
                'count': len(type_metrics['completion_scores'])
            }
        
        if type_metrics['timing_error_scores']:
            type_summary['timing_error'] = {
                'mean': np.mean(type_metrics['timing_error_scores']),
                'std': np.std(type_metrics['timing_error_scores']),
                'median': np.median(type_metrics['timing_error_scores']),
                'min': np.min(type_metrics['timing_error_scores']),
                'max': np.max(type_metrics['timing_error_scores']),
                'count': len(type_metrics['timing_error_scores'])
            }
        
        if type_metrics['nan_rates']:
            type_summary['nan_rate'] = {
                'mean': np.mean(type_metrics['nan_rates']),
                'std': np.std(type_metrics['nan_rates']),
                'median': np.median(type_metrics['nan_rates']),
                'min': np.min(type_metrics['nan_rates']),
                'max': np.max(type_metrics['nan_rates']),
                'count': len(type_metrics['nan_rates'])
            }
        
        overall_summary['by_model_type'][model_type] = type_summary
    
    # Calculate individual model rankings
    model_rankings = []
    for model_name, model_data in all_models.items():
        model_score = {
            'name': model_name,
            'type': model_data['type'],
            'task_count': len(model_data['tasks']),
            'scores': {}
        }
        
        # Calculate composite scores
        accuracy_scores = []
        cost_scores = []
        completion_scores = []
        
        for task_name, task_metrics in model_data['tasks'].items():
            if 'accuracy' in task_metrics:
                accuracy_scores.append(task_metrics['accuracy'])
            if 'mean_cost' in task_metrics:
                cost_scores.append(task_metrics['mean_cost'])
            if 'mean_completion' in task_metrics:
                completion_scores.append(task_metrics['mean_completion'])
        
        if accuracy_scores:
            model_score['scores']['mean_accuracy'] = np.mean(accuracy_scores)
        if cost_scores:
            model_score['scores']['mean_cost'] = np.mean(cost_scores)
        if completion_scores:
            model_score['scores']['mean_completion'] = np.mean(completion_scores)
        
        # Calculate composite performance score with task-weighted representation
        # Task representation: 60 accuracy-based models vs 30 completion-based models
        # Give production models 2x weight to balance representation
        composite_score = 0
        weight_sum = 0
        task_weight = 1.0  # Default weight
        
        # Determine task weight based on available metrics
        if 'mean_completion' in model_score['scores']:
            # Production task model - give 2x weight to balance representation
            task_weight = 2.0
        elif 'mean_accuracy' in model_score['scores']:
            # Comparison/Bisection task model - standard weight
            task_weight = 1.0
        
        # Primary performance metric (task-specific)
        if 'mean_accuracy' in model_score['scores']:
            # Accuracy for comparison/bisection tasks
            composite_score += model_score['scores']['mean_accuracy'] * 0.7 * task_weight
            weight_sum += 0.7 * task_weight
        elif 'mean_completion' in model_score['scores']:
            # Completion for production task (with 2x weight)
            composite_score += model_score['scores']['mean_completion'] * 0.7 * task_weight
            weight_sum += 0.7 * task_weight
        
        # Cost component (secondary, all tasks)
        if 'mean_cost' in model_score['scores'] and weight_sum > 0:
            # Invert cost (lower is better) and normalize to 0-1 range
            max_cost = 3.0  # Approximate max cost observed across all tasks
            normalized_cost = max(0, 1 - (model_score['scores']['mean_cost'] / max_cost))
            composite_score += normalized_cost * 0.3 * task_weight
            weight_sum += 0.3 * task_weight
        
        # Normalize by total weight to get final score
        if weight_sum > 0:
            model_score['composite_score'] = composite_score / weight_sum
            model_score['task_weight'] = task_weight
            model_score['score_components'] = {
                'has_accuracy': 'mean_accuracy' in model_score['scores'],
                'has_completion': 'mean_completion' in model_score['scores'],
                'has_cost': 'mean_cost' in model_score['scores'],
                'task_weight': task_weight,
                'weight_sum': weight_sum
            }
        else:
            model_score['composite_score'] = 0
            model_score['task_weight'] = task_weight
            model_score['score_components'] = {
                'has_accuracy': False,
                'has_completion': False,
                'has_cost': False,
                'task_weight': task_weight,
                'weight_sum': 0
            }
        
        model_rankings.append(model_score)
    
    # Sort by composite score
    model_rankings.sort(key=lambda x: x['composite_score'], reverse=True)
    overall_summary['individual_model_rankings'] = model_rankings
    
    return overall_summary


def generate_report(overall_results: Dict[str, Any], output_file: str = None) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        overall_results: Results from calculate_overall_performance
        output_file: Optional file to save the report
        
    Returns:
        Report string
    """
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "OVERALL PERFORMANCE ESTIMATION ACROSS ALL TASKS",
        "=" * 80,
        "",
        "This report aggregates performance across all interoception modeling tasks:",
        "- Interval Comparison",
        "- Interval Production", 
        "- Time Bisection",
        "",
        "Model Types: Standard (no_piezo), Piezo, Insula",
        "",
    ])
    
    # Task-level summary
    report_lines.extend([
        "TASK-LEVEL PERFORMANCE SUMMARY",
        "-" * 40,
    ])
    
    for task_name, task_perf in overall_results['by_task'].items():
        report_lines.append(f"\n{task_name.upper().replace('_', ' ')}:")
        for model_type in ['standard', 'piezo', 'insula']:
            if model_type in task_perf:
                perf = task_perf[model_type]
                report_lines.append(f"  {model_type.capitalize()}: {perf['mean']:.4f} ± {perf['std']:.4f}")
    
    report_lines.append("")
    
    # Model type summary
    report_lines.extend([
        "CROSS-TASK PERFORMANCE BY MODEL TYPE",
        "-" * 40,
    ])
    
    for model_type, type_data in overall_results['by_model_type'].items():
        report_lines.extend([
            f"\n{model_type.upper()} MODELS:",
            f"  Total models: {type_data['model_count']}",
        ])
        
        # Task coverage
        report_lines.append("  Task coverage:")
        for task, count in type_data['task_coverage'].items():
            report_lines.append(f"    {task}: {count} models")
        
        # Performance metrics
        if 'accuracy' in type_data:
            acc = type_data['accuracy']
            report_lines.append(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f} (n={acc['count']})")
            report_lines.append(f"    Range: [{acc['min']:.4f}, {acc['max']:.4f}], Median: {acc['median']:.4f}")
        
        if 'cost' in type_data:
            cost = type_data['cost']
            report_lines.append(f"  Mean Cost: {cost['mean']:.4f} ± {cost['std']:.4f} (n={cost['count']})")
            report_lines.append(f"    Range: [{cost['min']:.4f}, {cost['max']:.4f}], Median: {cost['median']:.4f}")
        
        if 'completion' in type_data:
            comp = type_data['completion']
            report_lines.append(f"  Completion Rate: {comp['mean']:.4f} ± {comp['std']:.4f} (n={comp['count']})")
            report_lines.append(f"    Range: [{comp['min']:.4f}, {comp['max']:.4f}], Median: {comp['median']:.4f}")
        
        if 'timing_error' in type_data:
            te = type_data['timing_error']
            report_lines.append(f"  Timing Error: {te['mean']:.4f} ± {te['std']:.4f} (n={te['count']})")
            report_lines.append(f"    Range: [{te['min']:.4f}, {te['max']:.4f}], Median: {te['median']:.4f}")
        
        if 'nan_rate' in type_data:
            nan = type_data['nan_rate']
            report_lines.append(f"  NaN Rate: {nan['mean']:.4f} ± {nan['std']:.4f} (n={nan['count']})")
            report_lines.append(f"    Range: [{nan['min']:.4f}, {nan['max']:.4f}], Median: {nan['median']:.4f}")
    
    # Model rankings
    report_lines.extend([
        "",
        "INDIVIDUAL MODEL RANKINGS",
        "-" * 40,
        "",
        "Top 20 models by composite performance score:",
        "(Task-weighted scoring: Primary metric 70%, Cost 30%)",
        "Production models get 2x weight to balance task representation",
        "",
        f"{'Rank':<5} {'Model Name':<25} {'Type':<10} {'Tasks':<6} {'Score':<8} {'Accuracy':<10} {'Completion':<12} {'Cost':<8}",
        "-" * 85,
    ])
    
    for i, model in enumerate(overall_results['individual_model_rankings'][:20]):
        rank = i + 1
        name = model['name']
        model_type = model['type']
        task_count = model['task_count']
        score = model['composite_score']
        
        accuracy = f"{model['scores'].get('mean_accuracy', 0):.4f}" if 'mean_accuracy' in model['scores'] else "N/A"
        completion = f"{model['scores'].get('mean_completion', 0):.4f}" if 'mean_completion' in model['scores'] else "N/A"
        cost = f"{model['scores'].get('mean_cost', 0):.4f}" if 'mean_cost' in model['scores'] else "N/A"
        
        report_lines.append(
            f"{rank:<5} {name:<25} {model_type:<10} {task_count:<6} {score:<8.4f} {accuracy:<10} {completion:<12} {cost:<8}"
        )
    
    # Performance comparison
    report_lines.extend([
        "",
        "",
        "PERFORMANCE COMPARISON SUMMARY",
        "-" * 40,
    ])
    
    # Find best performing model type for each metric
    best_accuracy = max(overall_results['by_model_type'].items(), 
                       key=lambda x: x[1].get('accuracy', {}).get('mean', 0) if 'accuracy' in x[1] else 0)
    
    best_completion = max(overall_results['by_model_type'].items(),
                         key=lambda x: x[1].get('completion', {}).get('mean', 0) if 'completion' in x[1] else 0)
    
    best_cost = min(overall_results['by_model_type'].items(),
                   key=lambda x: x[1].get('cost', {}).get('mean', float('inf')) if 'cost' in x[1] else float('inf'))
    
    best_nan = min(overall_results['by_model_type'].items(),
                  key=lambda x: x[1].get('nan_rate', {}).get('mean', float('inf')) if 'nan_rate' in x[1] else float('inf'))
    
    report_lines.extend([
        f"Best Accuracy: {best_accuracy[0].capitalize()} models ({best_accuracy[1].get('accuracy', {}).get('mean', 0):.4f})",
        f"Best Completion: {best_completion[0].capitalize()} models ({best_completion[1].get('completion', {}).get('mean', 0):.4f})",
        f"Best Cost: {best_cost[0].capitalize()} models ({best_cost[1].get('cost', {}).get('mean', 0):.4f})",
        f"Best NaN Rate: {best_nan[0].capitalize()} models ({best_nan[1].get('nan_rate', {}).get('mean', 0):.4f})",
        "",
        f"Overall Best Model: {overall_results['individual_model_rankings'][0]['name']} ({overall_results['individual_model_rankings'][0]['type']})",
        f"Composite Score: {overall_results['individual_model_rankings'][0]['composite_score']:.4f}",
        "",
    ])
    
    # Statistical significance summary
    if 'statistical_analysis' in overall_results:
        report_lines.extend([
            "",
            "STATISTICAL SIGNIFICANCE SUMMARY",
            "-" * 40,
        ])
        
        stat_results = overall_results['statistical_analysis']
        analyzer = overall_results.get('statistical_analyzer')
        
        for metric, result in stat_results.items():
            if 'error' in result:
                continue
                
            # Get the appropriate test result
            assumptions = result['assumptions']
            main_test = result['parametric'] if assumptions['use_parametric'] and result['parametric'] else result['nonparametric']
            main_p = main_test['p_value']
            
            # Determine significance level
            if main_p < 0.001:
                sig_level = "highly significant (p < 0.001)"
            elif main_p < 0.01:
                sig_level = "very significant (p < 0.01)"
            elif main_p < 0.05:
                sig_level = "significant (p < 0.05)"
            else:
                sig_level = "not significant (p ≥ 0.05)"
            
            metric_name = result.get('metric_name', metric.capitalize())
            report_lines.append(f"{metric_name}: {sig_level}")
            
            # Add significant pairwise comparisons
            if main_p < 0.05 and 'pairwise' in result:
                significant_pairs = []
                for comparison_key, pairwise_result in result['pairwise'].items():
                    if pairwise_result['corrected_p_value'] < 0.05:
                        group1, group2 = comparison_key.replace('_vs_', ' vs ').split(' vs ')
                        effect_interpretation = analyzer.interpret_effect_size(
                            pairwise_result['effect_size'], 
                            pairwise_result['effect_size_name']
                        ) if analyzer else "unknown"
                        significant_pairs.append(f"  {group1} vs {group2} (p={pairwise_result['corrected_p_value']:.3f}, {effect_interpretation} effect)")
                
                if significant_pairs:
                    report_lines.append("  Significant pairwise differences:")
                    report_lines.extend(significant_pairs)
        
        report_lines.append("")
    
    # Add note about statistical analysis
    if 'statistical_analysis' in overall_results:
        report_lines.extend([
            "NOTE: Detailed statistical analysis available in cross_task_statistical_analysis.txt",
            "Includes normality tests, effect sizes, and comprehensive pairwise comparisons.",
            "",
        ])
    
    # Footer
    report_lines.extend([
        "=" * 80,
        f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ])
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    return report


def main():
    """Main execution function."""
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading results from *_full_results directories...")
    all_results = load_all_results(base_dir)
    
    if not all_results:
        print("No results found! Make sure the *_full_results directories exist.")
        return
    
    print(f"Found results for {len(all_results)} tasks")
    
    print("Calculating overall performance metrics...")
    overall_results = calculate_overall_performance(all_results)
    
    # Perform statistical analysis
    print("Performing cross-task statistical analysis...")
    statistical_analyzer = CrossTaskStatisticalAnalyzer()
    model_type_data = statistical_analyzer.extract_cross_task_performance_data(all_results)
    statistical_results = statistical_analyzer.perform_cross_task_statistical_tests(model_type_data)
    
    # Add statistical results to overall results
    overall_results['statistical_analysis'] = statistical_results
    overall_results['statistical_analyzer'] = statistical_analyzer
    
    # Generate reports
    output_file = os.path.join(base_dir, "overall_performance_report.txt")
    report = generate_report(overall_results, output_file)
    
    # Generate statistical analysis report
    statistical_report = statistical_analyzer.generate_cross_task_statistical_report(statistical_results)
    statistical_output_file = os.path.join(base_dir, "cross_task_statistical_analysis.txt")
    with open(statistical_output_file, 'w') as f:
        f.write(statistical_report)
    print(f"Statistical analysis report saved to: {statistical_output_file}")
    
    # Generate performance plots
    print("Generating performance visualization plots...")
    plot_cross_task_performance_comparison(all_results, overall_results, base_dir)
    plot_composite_performance_scatter(overall_results, base_dir)
    
    # Save detailed results as JSON (excluding circular references)
    json_file = os.path.join(base_dir, "overall_performance_data.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Create a JSON-safe copy of results (exclude the analyzer object to avoid circular references)
    json_safe_results = {}
    for key, value in overall_results.items():
        if key != 'statistical_analyzer':  # Skip the analyzer object
            json_safe_results[key] = value
    
    # Deep convert the results
    try:
        json_str = json.dumps(json_safe_results, default=convert_numpy, indent=2)
        with open(json_file, 'w') as f:
            f.write(json_str)
        print(f"Detailed results saved to: {json_file}")
    except Exception as e:
        print(f"Warning: Could not save JSON results due to serialization error: {e}")
        print("Main reports were generated successfully.")
    
    # Print summary to console
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*50)
    
    for model_type, type_data in overall_results['by_model_type'].items():
        print(f"\n{model_type.upper()} MODELS:")
        if 'accuracy' in type_data:
            print(f"  Accuracy: {type_data['accuracy']['mean']:.4f} ± {type_data['accuracy']['std']:.4f}")
        if 'completion' in type_data:
            print(f"  Completion: {type_data['completion']['mean']:.4f} ± {type_data['completion']['std']:.4f}")
        if 'cost' in type_data:
            print(f"  Cost: {type_data['cost']['mean']:.4f} ± {type_data['cost']['std']:.4f}")
    
    print(f"\nBest Overall Model: {overall_results['individual_model_rankings'][0]['name']}")
    print(f"Model Type: {overall_results['individual_model_rankings'][0]['type']}")
    print(f"Composite Score: {overall_results['individual_model_rankings'][0]['composite_score']:.4f}")
    
    # Print statistical significance summary
    if 'statistical_analysis' in overall_results:
        print(f"\nStatistical Significance Summary:")
        print("="*40)
        stat_results = overall_results['statistical_analysis']
        analyzer = overall_results.get('statistical_analyzer')
        
        for metric, result in stat_results.items():
            if 'error' in result:
                continue
                
            assumptions = result['assumptions']
            main_test = result['parametric'] if assumptions['use_parametric'] and result['parametric'] else result['nonparametric']
            main_p = main_test['p_value']
            
            if main_p < 0.001:
                sig_level = "highly significant (p < 0.001)"
            elif main_p < 0.01:
                sig_level = "very significant (p < 0.01)"
            elif main_p < 0.05:
                sig_level = "significant (p < 0.05)"
            else:
                sig_level = "not significant (p ≥ 0.05)"
            
            metric_name = result.get('metric_name', metric.capitalize())
            print(f"{metric_name}: {sig_level}")
    
    print(f"\nFiles generated:")
    print(f"  Main report: {output_file}")
    if 'statistical_analysis' in overall_results:
        print(f"  Statistical analysis: {statistical_output_file}")
    print(f"  Cross-task performance plot: cross_task_performance_comparison.png")
    print(f"  Composite performance scatter: composite_performance_scatter.png")
    print(f"  Detailed data (JSON): overall_performance_data.json")


if __name__ == "__main__":
    main()
