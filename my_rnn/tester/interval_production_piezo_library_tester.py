"""
Interval Production Cardiac Library Comparison Tester
====================================================

Tests interval production piezo models across different cardiac libraries
to assess robustness and optimal cardiac data sources.
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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

from run import Runner
import tools
import task
import default
from real_cardiac_data import create_real_cardiac_data_for_task


class IntervalProductionCardiacLibraryTester:
    """Test interval production models across different cardiac libraries"""

    def __init__(self, base_results_dir="interval_production_cardiac_library_results"):
        self.base_results_dir = base_results_dir
        self.rule_name = 'interval_production'

        # Define available cardiac libraries
        self.cardiac_libraries = {
            "hr60_hrv0cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr60_hrv0cal",
            "hr60_hrv1cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr60_hrv1cal",
            "hr60_hrv3cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr60_hrv3cal",
            "hr90_hrv0cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr90_hrv0cal",
            "hr90_hrv1cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr90_hrv1cal",
            "hr90_hrv3cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr90_hrv3cal",
            "hr100_hrv0cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr100_hrv0cal",
            "hr100_hrv1cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr100_hrv1cal",
            "hr100_hrv3cal": "/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/piezo-wt/hr100_hrv3cal",
            # Add more libraries as needed
        }

        tools.mkdir_p(base_results_dir)

        print(f"Interval Production Cardiac Library Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Available cardiac libraries: {list(self.cardiac_libraries.keys())}")

    def get_test_conditions(self, num_conditions=10, use_dataset=True, include_ood=True):
        """Get test conditions for interval production - dataset sampling or predefined"""

        if use_dataset:
            # Load from your existing dataset (consistent with other testers)
            dataset_path = "enhanced_interval_datasets/interval_production_dataset.json"

            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)

                test_data = dataset.get('test', [])
                if not test_data:
                    # Fallback to train data sample
                    train_data = dataset.get('train', [])
                    if train_data:
                        random.seed(42)
                        test_data = random.sample(train_data, min(num_conditions, len(train_data)))

                selected = test_data[:num_conditions] if len(test_data) >= num_conditions else test_data

                # Add OOD conditions if requested
                if include_ood:
                    ood_conditions = [
                        {'prod_interval': 2600, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2500, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2612, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2625, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2637, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2650, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2663, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2675, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2686, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2700, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2725, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                        {'prod_interval': 2750, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                    ]
                    selected.extend(ood_conditions)

                print(
                    f"Using dataset conditions: {len(selected)} total ({len(selected) - (2 if include_ood else 0)} from dataset)")
                return selected

            else:
                print(f"Dataset not found at {dataset_path}, falling back to predefined conditions")

        # Fallback to predefined conditions (original behavior)
        conditions = [
            {'prod_interval': 1200, 'dly_interval': 1200},
            {'prod_interval': 1600, 'dly_interval': 1200},
            {'prod_interval': 2000, 'dly_interval': 1200},
            {'prod_interval': 2400, 'dly_interval': 1200},
        ]

        # Add OOD conditions if requested
        if include_ood:
            conditions.extend([
                {'prod_interval': 2600, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2500, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2612, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2625, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2637, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2650, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2663, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2675, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2686, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2700, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2725, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
                {'prod_interval': 2750, 'dly_interval': 1200, 'is_ood': True, 'ood_type': 'above_range'},
            ])

        print(f"Using predefined conditions: {len(conditions)} total")
        return conditions

    def test_model_with_single_library(self, model_dir, cardiac_library, test_conditions):
        """Test a single model with one specific cardiac library - FIXED VERSION"""

        # Verify model exists and has piezo
        analysis = tools.analyze_model_directory(model_dir)
        if not analysis['exists']:
            print(f"Model directory not found: {model_dir}")
            return None

        if not analysis['use_piezo']:
            print(f"Model {model_dir} is not a piezo model - skipping")
            return None

        model_name = os.path.basename(model_dir)

        print(f"  Testing {model_name} with {cardiac_library}")

        # Initialize runner
        runner = Runner(
            rule_name=self.rule_name,
            model_dir=model_dir,
            is_cuda=False,
            noise_on=False
        )

        results = {
            'model_dir': model_dir,
            'model_name': model_name,
            'cardiac_library': cardiac_library,
            'test_results': [],
            'performance_summary': {}
        }

        # Test each condition with the specified cardiac library
        for i, condition in enumerate(test_conditions):
            print(f"    Condition {i + 1}/{len(test_conditions)}: prod={condition['prod_interval']}ms")

            try:
                # FIXED: Use the same cardiac data loading method as the working tester
                # Generate cardiac data from the specific library via TaskDataset approach
                task_duration_ms = condition['prod_interval'] + condition['dly_interval'] + 1000

                # Get model info for TaskDataset
                import default
                hp = default.get_default_hp(rule_name=self.rule_name, use_piezo=True)
                hp.update(runner.get_model_info())

                # Create a temporary modified hp with the specific cardiac library
                # We need to monkey-patch the cardiac library for this test
                original_create_real_cardiac_data = None
                try:
                    # Import the function to patch it
                    from real_cardiac_data import create_real_cardiac_data_for_task

                    # Store original function
                    original_create_real_cardiac_data = create_real_cardiac_data_for_task

                    # Create a wrapper that uses our specific library
                    def library_specific_cardiac_data(task_duration_ms, dt=20.0, slice_size=20, **kwargs):
                        return original_create_real_cardiac_data(
                            task_duration_ms=task_duration_ms,
                            dt=dt,
                            slice_size=slice_size,
                            cardiac_library=cardiac_library
                        )

                    # Temporarily replace the function
                    import real_cardiac_data
                    real_cardiac_data.create_real_cardiac_data_for_task = library_specific_cardiac_data

                    # Use TaskDataset approach (same as working tester)
                    from dataset import TaskDataset
                    ds = TaskDataset(self.rule_name, hp, mode="test")
                    sample = ds[0]

                    # Add cardiac data to condition
                    condition_with_cardiac = condition.copy()
                    condition_with_cardiac['hb_sequence'] = sample['hb_sequence']

                finally:
                    # Restore original function
                    if original_create_real_cardiac_data:
                        real_cardiac_data.create_real_cardiac_data_for_task = original_create_real_cardiac_data

                # Run the model
                trial, train_stepper = runner.run(**condition_with_cardiac)

                # Extract results
                outputs = train_stepper.outputs.detach().cpu().numpy()
                states = [state.detach().cpu().numpy() for state in train_stepper.state_collector]
                cost = train_stepper.cost.item()

                # FIXED: Use the same NaN checking logic as the working tester
                # Only check for truly problematic NaN values, not performance artifacts
                has_nan_outputs = np.isnan(outputs).any()
                has_nan_states = any(np.isnan(state).any() for state in states)
                has_nan_cost = np.isnan(cost)

                # Calculate performance
                performance = self._calculate_interval_production_performance(trial, outputs)

                # FIXED: Don't mark as has_nan just because of infinite timing errors
                # Only mark as has_nan if the completion calculation itself failed
                performance_has_critical_nan = False
                if 'mean_completion' in performance:
                    if np.isnan(performance['mean_completion']):
                        performance_has_critical_nan = True

                # FIXED: Only mark as has_nan for truly critical failures
                has_nan = has_nan_outputs or has_nan_states or has_nan_cost or performance_has_critical_nan

                # Get cardiac metadata if available
                cardiac_metadata = {}
                if 'hb_sequence' in condition_with_cardiac:
                    # Try to extract metadata from the cardiac data generation
                    try:
                        cardiac_data = create_real_cardiac_data_for_task(
                            task_duration_ms=task_duration_ms,
                            dt=runner.hp.get('dt', 20),
                            slice_size=runner.hp.get('heartbeat_slice_size', 20),
                            cardiac_library=cardiac_library
                        )
                        cardiac_metadata = {
                            'heart_rate': cardiac_data['heart_rate'],
                            'r_peak_count': len(cardiac_data['r_peak_times']),
                            'total_duration_s': cardiac_data['total_duration_s'],
                            'pressure_stats': {
                                'min': cardiac_data['min_pressure'],
                                'max': cardiac_data['max_pressure'],
                                'mean': cardiac_data['pressure_mean'],
                                'std': cardiac_data['pressure_std']
                            }
                        }
                    except Exception as e:
                        print(f"    Warning: Could not extract cardiac metadata: {e}")
                        cardiac_metadata = {'error': str(e)}

                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'performance': performance,
                    'cost': cost,
                    'has_nan': has_nan,
                    'cardiac_library': cardiac_library,
                    'cardiac_metadata': cardiac_metadata
                }

                results['test_results'].append(test_result)

            except Exception as e:
                print(f"    Error in condition {i + 1}: {e}")
                # Add a failed result to maintain index consistency
                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'performance': {'mean_completion': np.nan, 'mean_timing_error': np.inf},
                    'cost': np.nan,
                    'has_nan': True,
                    'cardiac_library': cardiac_library,
                    'cardiac_metadata': {'error': str(e)}
                }
                results['test_results'].append(test_result)
                continue

        # FIXED: Use the same aggregation logic as the working tester
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance_fixed(results['test_results'])

        return results

    def _summarize_performance_fixed(self, test_results):
        """FIXED: Summarize performance using the same logic as the working tester"""

        if not test_results:
            return {}

        # FIXED: Use the same filtering logic as interval_production_tester.py
        # Don't filter by has_nan - only filter by actual NaN values in the metrics
        completions = [r['performance']['mean_completion'] for r in test_results
                       if 'mean_completion' in r['performance'] and not np.isnan(r['performance']['mean_completion'])]

        timing_errors = [r['performance']['mean_timing_error'] for r in test_results
                         if
                         'mean_timing_error' in r['performance'] and np.isfinite(r['performance']['mean_timing_error'])]

        costs = [r['cost'] for r in test_results if not np.isnan(r['cost']) and np.isfinite(r['cost'])]

        # Calculate summary with better error handling
        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(completions),
            'nan_rate': (len(test_results) - len(completions)) / len(test_results) if test_results else 0,
            'mean_completion': np.mean(completions) if completions else 0,  # FIXED: Don't use nanmean on empty list
            'std_completion': np.std(completions) if completions else 0,
            'mean_timing_error': np.mean(timing_errors) if timing_errors else np.inf,
            'std_timing_error': np.std(timing_errors) if timing_errors else 0,
            'mean_cost': np.mean(costs) if costs else np.inf
        }

        # Debug output to see what's happening
        print(f"    Summary: {len(completions)} valid completions out of {len(test_results)} conditions")
        if completions:
            print(f"    Mean completion: {summary['mean_completion']:.4f}")
        else:
            print(f"    No valid completions found!")

        return summary

    def _generate_written_report(self, all_results):
        """Generate comprehensive written report for cardiac library comparison"""

        report_path = os.path.join(self.base_results_dir, "cardiac_library_comparison_report.txt")

        # Prepare data for analysis
        performance_data = []
        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    performance_data.append({
                        'model': model_name,
                        'library': lib_name,
                        'mean_completion': lib_results['performance_summary']['mean_completion'],
                        'std_completion': lib_results['performance_summary']['std_completion'],
                        'mean_timing_error': lib_results['performance_summary']['mean_timing_error'],
                        'std_timing_error': lib_results['performance_summary']['std_timing_error'],
                        'mean_cost': lib_results['performance_summary']['mean_cost'],
                        'nan_rate': lib_results['performance_summary']['nan_rate'],
                        'num_conditions': lib_results['performance_summary']['num_conditions'],
                        'num_valid': lib_results['performance_summary']['num_valid']
                    })

        df = pd.DataFrame(performance_data)

        with open(report_path, 'w') as f:
            f.write("INTERVAL PRODUCTION CARDIAC LIBRARY COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Models Tested: {len(all_results)}\n")
            f.write(f"Cardiac Libraries: {len(self.cardiac_libraries)}\n")
            f.write(f"Total Comparisons: {len(performance_data)}\n")
            f.write(
                f"Overall Mean Completion: {df['mean_completion'].mean():.4f} ± {df['mean_completion'].std():.4f}\n")
            f.write(f"Performance Range: {df['mean_completion'].min():.4f} - {df['mean_completion'].max():.4f}\n")

            # Best and worst performing combinations
            best_combo = df.loc[df['mean_completion'].idxmax()]
            worst_combo = df.loc[df['mean_completion'].idxmin()]
            f.write(
                f"Best Combination: {best_combo['model']} + {best_combo['library']} ({best_combo['mean_completion']:.4f})\n")
            f.write(
                f"Worst Combination: {worst_combo['model']} + {worst_combo['library']} ({worst_combo['mean_completion']:.4f})\n\n")

            # Library Rankings
            f.write("CARDIAC LIBRARY PERFORMANCE RANKINGS\n")
            f.write("-" * 45 + "\n")
            library_stats = df.groupby('library').agg({
                'mean_completion': ['mean', 'std', 'count'],
                'mean_timing_error': 'mean',
                'nan_rate': 'mean'
            }).round(4)

            library_means = df.groupby('library')['mean_completion'].mean().sort_values(ascending=False)

            f.write("Ranked by Mean Completion Ratio (Higher = Better):\n\n")
            for rank, (lib_name, mean_completion) in enumerate(library_means.items(), 1):
                lib_data = df[df['library'] == lib_name]
                std_completion = lib_data['mean_completion'].std()
                count = len(lib_data)
                mean_error = lib_data['mean_timing_error'].mean()
                nan_rate = lib_data['nan_rate'].mean()

                f.write(f"{rank:2d}. {lib_name:15s} | Completion: {mean_completion:.4f} ± {std_completion:.4f} "
                        f"| Error: {mean_error:.1f} | NaN Rate: {nan_rate:.1%} | n={count}\n")

            f.write(f"\nLibrary Effect Size (std of library means): {library_means.std():.4f}\n")

            # Determine if library choice matters significantly
            library_cv = library_means.std() / library_means.mean()
            f.write(f"Library Coefficient of Variation: {library_cv:.4f}")
            if library_cv > 0.1:
                f.write(" (HIGH - Library choice significantly affects performance)\n")
            elif library_cv > 0.05:
                f.write(" (MODERATE - Library choice moderately affects performance)\n")
            else:
                f.write(" (LOW - Library choice has minimal effect on performance)\n")
            f.write("\n")

            # Model Consistency Analysis
            f.write("MODEL CONSISTENCY ANALYSIS\n")
            f.write("-" * 35 + "\n")
            model_stats = df.groupby('model').agg({
                'mean_completion': ['mean', 'std'],
                'library': 'count'
            }).round(4)

            f.write("Models ranked by consistency (Lower Std = More Consistent across libraries):\n\n")
            model_consistency = df.groupby('model')['mean_completion'].std().sort_values()

            for rank, (model_name, std_completion) in enumerate(model_consistency.items(), 1):
                model_data = df[df['model'] == model_name]
                mean_completion = model_data['mean_completion'].mean()
                count = len(model_data)
                min_perf = model_data['mean_completion'].min()
                max_perf = model_data['mean_completion'].max()

                f.write(f"{rank:2d}. {model_name:20s} | Mean: {mean_completion:.4f} | Std: {std_completion:.4f} "
                        f"| Range: [{min_perf:.4f}, {max_perf:.4f}] | n={count}\n")

            f.write("\n")

            # Detailed Library Analysis
            f.write("DETAILED LIBRARY ANALYSIS\n")
            f.write("-" * 30 + "\n")

            for lib_name in sorted(self.cardiac_libraries.keys()):
                lib_data = df[df['library'] == lib_name]
                if len(lib_data) == 0:
                    continue

                f.write(f"\n{lib_name.upper()}:\n")
                f.write(f"  Models Tested: {len(lib_data)}\n")
                f.write(
                    f"  Mean Completion: {lib_data['mean_completion'].mean():.4f} ± {lib_data['mean_completion'].std():.4f}\n")
                f.write(
                    f"  Completion Range: [{lib_data['mean_completion'].min():.4f}, {lib_data['mean_completion'].max():.4f}]\n")
                f.write(
                    f"  Mean Timing Error: {lib_data['mean_timing_error'].mean():.1f} ± {lib_data['mean_timing_error'].std():.1f} steps\n")
                f.write(f"  Mean Cost: {lib_data['mean_cost'].mean():.4f}\n")
                f.write(f"  Average NaN Rate: {lib_data['nan_rate'].mean():.1%}\n")

                # Best and worst models for this library
                best_model = lib_data.loc[lib_data['mean_completion'].idxmax()]
                worst_model = lib_data.loc[lib_data['mean_completion'].idxmin()]
                f.write(f"  Best Model: {best_model['model']} ({best_model['mean_completion']:.4f})\n")
                f.write(f"  Worst Model: {worst_model['model']} ({worst_model['mean_completion']:.4f})\n")

            # Detailed Model Analysis
            f.write("\n\nDETAILED MODEL ANALYSIS\n")
            f.write("-" * 28 + "\n")

            for model_name in sorted(df['model'].unique()):
                model_data = df[df['model'] == model_name]

                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Libraries Tested: {len(model_data)}\n")
                f.write(
                    f"  Mean Completion: {model_data['mean_completion'].mean():.4f} ± {model_data['mean_completion'].std():.4f}\n")
                f.write(
                    f"  Completion Range: [{model_data['mean_completion'].min():.4f}, {model_data['mean_completion'].max():.4f}]\n")
                f.write(
                    f"  Mean Timing Error: {model_data['mean_timing_error'].mean():.1f} ± {model_data['mean_timing_error'].std():.1f} steps\n")
                f.write(f"  Average NaN Rate: {model_data['nan_rate'].mean():.1%}\n")

                # Best and worst libraries for this model
                best_library = model_data.loc[model_data['mean_completion'].idxmax()]
                worst_library = model_data.loc[model_data['mean_completion'].idxmin()]
                f.write(f"  Best Library: {best_library['library']} ({best_library['mean_completion']:.4f})\n")
                f.write(f"  Worst Library: {worst_library['library']} ({worst_library['mean_completion']:.4f})\n")
                f.write(f"  Library Sensitivity: {model_data['mean_completion'].std():.4f} (lower = more robust)\n")

            # Statistical Analysis
            f.write("\n\nSTATISTICAL ANALYSIS\n")
            f.write("-" * 25 + "\n")

            # ANOVA-like analysis
            total_variance = df['mean_completion'].var()
            between_library_variance = df.groupby('library')['mean_completion'].mean().var()
            between_model_variance = df.groupby('model')['mean_completion'].mean().var()

            f.write(f"Total Performance Variance: {total_variance:.6f}\n")
            f.write(
                f"Between-Library Variance: {between_library_variance:.6f} ({between_library_variance / total_variance:.1%} of total)\n")
            f.write(
                f"Between-Model Variance: {between_model_variance:.6f} ({between_model_variance / total_variance:.1%} of total)\n")

            # Correlation analysis
            f.write(f"\nPerformance Correlations:\n")
            f.write(f"  Completion vs Timing Error: {df['mean_completion'].corr(df['mean_timing_error']):.3f}\n")
            f.write(f"  Completion vs Cost: {df['mean_completion'].corr(df['mean_cost']):.3f}\n")
            f.write(f"  Completion vs NaN Rate: {df['mean_completion'].corr(df['nan_rate']):.3f}\n")

            # Top and bottom performing combinations
            f.write(f"\n\nTOP 10 MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 40 + "\n")
            top_combinations = df.nlargest(10, 'mean_completion')
            for i, row in top_combinations.iterrows():
                f.write(
                    f"{row.name + 1:2d}. {row['model']:20s} + {row['library']:15s} | {row['mean_completion']:.4f}\n")

            f.write(f"\n\nBOTTOM 10 MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 42 + "\n")
            bottom_combinations = df.nsmallest(10, 'mean_completion')
            for i, row in bottom_combinations.iterrows():
                f.write(
                    f"{len(df) - row.name:2d}. {row['model']:20s} + {row['library']:15s} | {row['mean_completion']:.4f}\n")

            # Recommendations
            f.write(f"\n\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")

            best_overall_library = library_means.index[0]
            most_consistent_library = df.groupby('library')['mean_completion'].std().idxmin()
            most_robust_model = model_consistency.index[0]
            best_overall_model = df.groupby('model')['mean_completion'].mean().idxmax()

            f.write(f"1. BEST OVERALL LIBRARY: {best_overall_library}\n")
            f.write(f"   Highest mean performance across all models\n\n")

            f.write(f"2. MOST CONSISTENT LIBRARY: {most_consistent_library}\n")
            f.write(f"   Lowest performance variance across models\n\n")

            f.write(f"3. MOST ROBUST MODEL: {most_robust_model}\n")
            f.write(f"   Lowest performance variance across libraries\n\n")

            f.write(f"4. BEST PERFORMING MODEL: {best_overall_model}\n")
            f.write(f"   Highest mean performance across all libraries\n\n")

            if library_cv > 0.1:
                f.write(f"5. LIBRARY SELECTION IS CRITICAL:\n")
                f.write(f"   High coefficient of variation ({library_cv:.3f}) indicates\n")
                f.write(f"   cardiac library choice significantly impacts performance.\n")
                f.write(f"   Recommend using {best_overall_library} for optimal results.\n\n")
            else:
                f.write(f"5. LIBRARY SELECTION IS NOT CRITICAL:\n")
                f.write(f"   Low coefficient of variation ({library_cv:.3f}) indicates\n")
                f.write(f"   models are robust to cardiac library choice.\n\n")

            # Technical Notes
            f.write(f"TECHNICAL NOTES\n")
            f.write("-" * 20 + "\n")
            f.write(f"- Completion ratio: 1.0 = perfect timing, lower = worse timing\n")
            f.write(f"- Timing error: Steps deviation from target interval\n")
            f.write(f"- NaN rate: Proportion of trials producing invalid outputs\n")
            f.write(f"- Library CV: Coefficient of variation of library means\n")
            f.write(f"- Model sensitivity: Standard deviation across libraries\n")
            f.write(
                f"- Total conditions tested per model-library pair: {df['num_conditions'].iloc[0] if len(df) > 0 else 'N/A'}\n")

            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")

        print(f"Comprehensive written report saved to: {report_path}")
        return report_path

    def _calculate_interval_production_performance(self, trial, outputs):
        """Calculate interval production performance metrics"""

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

        return {
            'mean_timing_error': float(np.mean(response_errors[np.isfinite(response_errors)])),
            'timing_errors': response_errors.tolist(),
            'target_intervals': target_steps.tolist(),
            'completion_ratios': completions,
            'mean_completion': float(np.nanmean(completions)),
            'go_start_time': go_start
        }

    def _summarize_performance(self, test_results):
        """Summarize performance across test conditions"""

        if not test_results:
            return {}

        # Extract completion ratios and timing errors
        completions = [r['performance']['mean_completion'] for r in test_results
                       if not r['has_nan'] and not np.isnan(r['performance']['mean_completion'])]

        timing_errors = [r['performance']['mean_timing_error'] for r in test_results
                         if not r['has_nan'] and np.isfinite(r['performance']['mean_timing_error'])]

        costs = [r['cost'] for r in test_results if not r['has_nan'] and np.isfinite(r['cost'])]

        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(completions),
            'nan_rate': (len(test_results) - len(completions)) / len(test_results),
            'mean_completion': np.mean(completions) if completions else 0,
            'std_completion': np.std(completions) if completions else 0,
            'mean_timing_error': np.mean(timing_errors) if timing_errors else np.inf,
            'std_timing_error': np.std(timing_errors) if timing_errors else 0,
            'mean_cost': np.mean(costs) if costs else np.inf
        }

        return summary

    def test_model_across_libraries(self, model_dir, test_conditions=None):
        """Test a single model across all available cardiac libraries"""

        if test_conditions is None:
            test_conditions = self.get_test_conditions()

        model_name = os.path.basename(model_dir)
        print(f"\nTesting {model_name} across {len(self.cardiac_libraries)} cardiac libraries")

        results = {
            'model_dir': model_dir,
            'model_name': model_name,
            'library_results': {},
            'comparison_summary': {}
        }

        # Test with each cardiac library
        for library_name in self.cardiac_libraries.keys():
            library_result = self.test_model_with_single_library(
                model_dir, library_name, test_conditions
            )

            if library_result:
                results['library_results'][library_name] = library_result
                print(
                    f"    Completed {library_name}: mean_completion = {library_result['performance_summary']['mean_completion']:.4f}")
            else:
                print(f"    Failed {library_name}")

        # Generate comparison summary
        results['comparison_summary'] = self._generate_library_comparison(results['library_results'])

        return results

    def _generate_library_comparison(self, library_results):
        """Generate comparison statistics across libraries"""

        valid_libraries = {lib: results for lib, results in library_results.items()
                           if results and 'performance_summary' in results}

        if len(valid_libraries) < 2:
            return {'error': 'Need at least 2 valid libraries for comparison'}

        # Extract performance metrics
        library_performances = {}
        library_errors = {}
        library_costs = {}

        for lib_name, lib_results in valid_libraries.items():
            summary = lib_results['performance_summary']
            library_performances[lib_name] = summary['mean_completion']
            library_errors[lib_name] = summary['mean_timing_error']
            library_costs[lib_name] = summary['mean_cost']

        # Calculate comparison statistics
        performances = list(library_performances.values())

        comparison = {
            'library_performances': library_performances,
            'library_timing_errors': library_errors,
            'library_costs': library_costs,
            'overall_mean_completion': np.mean(performances),
            'overall_std_completion': np.std(performances),
            'completion_range': max(performances) - min(performances),
            'best_library': max(library_performances.items(), key=lambda x: x[1])[0],
            'worst_library': min(library_performances.items(), key=lambda x: x[1])[0],
            'performance_coefficient_of_variation': np.std(performances) / np.mean(performances) if np.mean(
                performances) > 0 else np.inf
        }

        return comparison

    def test_multiple_models_across_libraries(self, model_directories, num_conditions=15, use_dataset=True,
                                              include_ood=True):
        """Test multiple models across all libraries"""

        # Get test conditions
        test_conditions = self.get_test_conditions(num_conditions, use_dataset, include_ood)

        # Filter for piezo models only
        piezo_models = []
        for model_info in model_directories:
            if len(model_info) == 3:
                model_dir, rule_name, run_name = model_info
            else:
                model_dir, rule_name = model_info
                run_name = os.path.basename(model_dir)

            analysis = tools.analyze_model_directory(model_dir)
            if analysis['use_piezo']:
                piezo_models.append((model_dir, rule_name, run_name))

        print(f"Found {len(piezo_models)} piezo models to test")

        all_results = []

        for model_dir, rule_name, run_name in piezo_models:
            print(f"\n{'=' * 60}")
            print(f"Testing model: {run_name}")
            print(f"{'=' * 60}")

            results = self.test_model_across_libraries(model_dir, test_conditions)

            if results and results['library_results']:
                results['model_name'] = run_name
                all_results.append(results)

                # Save individual results
                results_path = os.path.join(self.base_results_dir, f"{run_name}_cardiac_library_test.pkl")
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Saved results: {results_path}")


        if all_results:
            self._generate_cross_model_analysis(all_results)
            self._create_comparison_plots(all_results)

            # ADD THIS LINE:
            self._generate_written_report(all_results)  # NEW: Generate comprehensive written report

            # Save consolidated results
            summary_path = os.path.join(self.base_results_dir, "all_cardiac_library_test_results.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

        return all_results

        return all_results

    def _generate_cross_model_analysis(self, all_results):
        """Generate analysis across all models and libraries"""

        print(f"\nGenerating cross-model analysis...")

        # Create performance matrix: models x libraries
        performance_data = []
        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    performance_data.append({
                        'model': model_name,
                        'library': lib_name,
                        'mean_completion': lib_results['performance_summary']['mean_completion'],
                        'mean_timing_error': lib_results['performance_summary']['mean_timing_error'],
                        'mean_cost': lib_results['performance_summary']['mean_cost'],
                        'nan_rate': lib_results['performance_summary']['nan_rate']
                    })

        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data)

        # Generate summary statistics
        analysis = {
            'overall_stats': {
                'num_models': len(all_results),
                'num_libraries': len(self.cardiac_libraries),
                'total_comparisons': len(performance_data),
                'mean_completion_overall': df['mean_completion'].mean(),
                'std_completion_overall': df['mean_completion'].std()
            },
            'library_rankings': df.groupby('library')['mean_completion'].agg(['mean', 'std', 'count']).to_dict(),
            'model_consistency': df.groupby('model')['mean_completion'].agg(['mean', 'std', 'count']).to_dict(),
            'library_effect_size': df.groupby('library')['mean_completion'].mean().std(),
            'best_combinations': df.nlargest(5, 'mean_completion')[['model', 'library', 'mean_completion']].to_dict(
                'records'),
            'worst_combinations': df.nsmallest(5, 'mean_completion')[['model', 'library', 'mean_completion']].to_dict(
                'records')
        }

        # Save analysis
        analysis_path = os.path.join(self.base_results_dir, "cross_model_library_analysis.json")
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2)

        print(f"Cross-model analysis saved to: {analysis_path}")

        # Print summary
        print(f"\nLIBRARY RANKINGS (by mean completion):")
        library_means = df.groupby('library')['mean_completion'].mean().sort_values(ascending=False)
        for lib, perf in library_means.items():
            print(f"  {lib}: {perf:.4f}")

        print(f"\nMODEL CONSISTENCY (by completion std - lower is more consistent):")
        model_stds = df.groupby('model')['mean_completion'].std().sort_values()
        for model, std in model_stds.items():
            print(f"  {model}: {std:.4f}")

    def _create_comparison_plots(self, all_results):
        """Create visualization plots for the comparison"""

        # Prepare data for plotting
        models = []
        libraries = []
        completions = []

        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    models.append(model_name)
                    libraries.append(lib_name)
                    completions.append(lib_results['performance_summary']['mean_completion'])

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Model': models,
            'Library': libraries,
            'Completion': completions
        })

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Interval Production: Cardiac Library Comparison', fontsize=16)

        # Plot 1: Library performance boxplot
        ax1 = axes[0, 0]
        libraries_unique = df['Library'].unique()
        library_data = [df[df['Library'] == lib]['Completion'].values for lib in libraries_unique]
        ax1.boxplot(library_data, labels=libraries_unique)
        ax1.set_title('Performance Distribution by Library')
        ax1.set_ylabel('Mean Completion Ratio')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Model consistency
        ax2 = axes[0, 1]
        model_means = df.groupby('Model')['Completion'].mean()
        model_stds = df.groupby('Model')['Completion'].std()
        ax2.scatter(model_means, model_stds, alpha=0.7, s=100)
        ax2.set_xlabel('Mean Completion Ratio')
        ax2.set_ylabel('Std Completion Ratio')
        ax2.set_title('Model Consistency (lower std = more consistent)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Heatmap of model x library performance
        ax3 = axes[1, 0]
        pivot_data = df.pivot(index='Model', columns='Library', values='Completion')
        im = ax3.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels(pivot_data.columns, rotation=45)
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels(pivot_data.index)
        ax3.set_title('Model × Library Performance Heatmap')
        plt.colorbar(im, ax=ax3, label='Mean Completion Ratio')

        # Plot 4: Library effect size
        ax4 = axes[1, 1]
        library_means = df.groupby('Library')['Completion'].mean().sort_values(ascending=False)
        library_stds = df.groupby('Library')['Completion'].std().reindex(library_means.index)
        x_pos = range(len(library_means))
        ax4.bar(x_pos, library_means.values, yerr=library_stds.values,
                capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(library_means.index, rotation=45)
        ax4.set_ylabel('Mean Completion Ratio')
        ax4.set_title('Library Performance Ranking')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, "cardiac_library_comparison_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plots saved to: {plot_path}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def discover_production_models(self):
        """Discover trained interval production models"""
        model_dirs = []

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
                    if 'run_' in item or item.startswith('w2_') or item.isdigit() or 'piezo' in item:
                        best_model_path = self._find_best_model_checkpoint(item_path)
                        if best_model_path:
                            model_dirs.append((best_model_path, rule_name, item))
                    elif self._has_valid_model(item_path):
                        model_dirs.append((item_path, rule_name, item))
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
            # Try to find the best checkpoint based on timing error from individual checkpoint logs
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


def main():
    """Run the cardiac library comparison test"""

    # Initialize the tester
    tester = IntervalProductionCardiacLibraryTester()

    # Discover interval production models
    model_directories = tester.discover_production_models()

    if not model_directories:
        print("No interval production models found!")
        return

    print(f"Found {len(model_directories)} interval production models")

    # Run the comparison across libraries
    results = tester.test_multiple_models_across_libraries(model_directories)

    print(f"\nCardiac library comparison complete!")
    print(f"Results saved to: {tester.base_results_dir}")

    return results


if __name__ == "__main__":
    main()