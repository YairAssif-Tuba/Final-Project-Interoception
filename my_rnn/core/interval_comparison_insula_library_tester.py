"""
Interval Comparison Insula Library Comparison Tester
====================================================

Tests interval comparison insula models across different cardiac libraries
to assess robustness and optimal cardiac data sources.

Based on interval_comparison_library_tester.py but adapted for insula models.
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


class IntervalComparisonInsulaLibraryTester:
    """Test interval comparison insula models across different cardiac libraries"""

    def __init__(self, base_results_dir="interval_comparison_insula_library_results"):
        self.base_results_dir = base_results_dir
        self.rule_name = 'interval_comparison'

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

        print(f"Interval Comparison Insula Library Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Available cardiac libraries: {list(self.cardiac_libraries.keys())}")

    def get_test_conditions(self, num_conditions=10, use_dataset=True, include_ood=True):
        """Get test conditions for interval comparison - dataset sampling or predefined"""

        if use_dataset:
            # Load from interval comparison dataset
            dataset_path = "enhanced_interval_datasets/interval_comparison_dataset.json"

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
                        {'prod_interval1': 2600, 'prod_interval2': 1400, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2500, 'prod_interval2': 1300, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2700, 'prod_interval2': 1200, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 1300, 'prod_interval2': 2600, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 1400, 'prod_interval2': 2500, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 1200, 'prod_interval2': 2700, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2650, 'prod_interval2': 2600, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2750, 'prod_interval2': 2650, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2800, 'prod_interval2': 1300, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 1300, 'prod_interval2': 2800, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 2900, 'prod_interval2': 2850, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                        {'prod_interval1': 3000, 'prod_interval2': 1400, 'dly_interval': 1200, 'is_ood': True,
                         'ood_type': 'above_range'},
                    ]
                    selected.extend(ood_conditions)

                print(
                    f"Using dataset conditions: {len(selected)} total ({len(selected) - (12 if include_ood else 0)} from dataset)")
                return selected

            else:
                print(f"Dataset not found at {dataset_path}, falling back to predefined conditions")

        # Fallback to predefined conditions
        conditions = [
            {'prod_interval1': 1200, 'prod_interval2': 1400, 'dly_interval': 1200},
            {'prod_interval1': 1600, 'prod_interval2': 1800, 'dly_interval': 1200},
            {'prod_interval1': 2000, 'prod_interval2': 1600, 'dly_interval': 1200},
            {'prod_interval1': 2400, 'prod_interval2': 2000, 'dly_interval': 1200},
        ]

        # Add OOD conditions if requested
        if include_ood:
            conditions.extend([
                {'prod_interval1': 2600, 'prod_interval2': 1400, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2500, 'prod_interval2': 1300, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2700, 'prod_interval2': 1200, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 1300, 'prod_interval2': 2600, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 1400, 'prod_interval2': 2500, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 1200, 'prod_interval2': 2700, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2650, 'prod_interval2': 2600, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2750, 'prod_interval2': 2650, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2800, 'prod_interval2': 1300, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 1300, 'prod_interval2': 2800, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 2900, 'prod_interval2': 2850, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'prod_interval1': 3000, 'prod_interval2': 1400, 'dly_interval': 1200, 'is_ood': True,
                 'ood_type': 'above_range'},
            ])

        print(f"Using predefined conditions: {len(conditions)} total")
        return conditions

    def test_model_with_single_library(self, model_dir, cardiac_library, test_conditions):
        """Test a single model with one specific cardiac library - INSULA VERSION"""

        # Verify model exists and is insula or baseline (not piezo)
        analysis = tools.analyze_model_directory(model_dir)
        if not analysis['exists']:
            print(f"Model directory not found: {model_dir}")
            return None

        # Use the same filtering logic as model discovery
        if not self._is_insula_model(model_dir):
            print(f"Model {model_dir} is not an insula model - skipping")
            return None

        model_name = os.path.basename(model_dir)

        print(f"  Testing {model_name} with {cardiac_library}")

        # Initialize runner
        runner = Runner(
            rule_name=self.rule_name,
            model_dir=model_dir,
            is_cuda=True,
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
            print(
                f"    Condition {i + 1}/{len(test_conditions)}: I1={condition['prod_interval1']}ms, I2={condition['prod_interval2']}ms")

            try:
                # Calculate task duration for this condition (needed for both insula and metadata)
                task_duration_ms = condition['prod_interval1'] + condition['prod_interval2'] + condition[
                    'dly_interval'] + 1000
                
                # Get model info to determine if this is an insula model
                model_info = runner.get_model_info()
                use_insula = model_info.get("use_insula", False)
                
                # Prepare condition for testing
                condition_with_cardiac = condition.copy()
                
                # Only load cardiac data for insula models
                if use_insula:
                    try:
                        # Get model info for TaskDataset
                        import default
                        hp = default.get_default_hp(rule_name=self.rule_name, 
                                                   use_insula=True)
                        hp.update(model_info)

                        # Create a temporary modified hp with the specific cardiac library
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

                            # Use TaskDataset approach
                            from dataset import TaskDataset
                            ds = TaskDataset(self.rule_name, hp, mode="test")
                            sample = ds[0]

                            # Add cardiac data to condition
                            condition_with_cardiac['hb_sequence'] = sample['hb_sequence']
                            print(f"    Loaded library-specific cardiac data for insula model")

                        finally:
                            # Restore original function
                            if original_create_real_cardiac_data:
                                real_cardiac_data.create_real_cardiac_data_for_task = original_create_real_cardiac_data
                    except Exception as fallback_error:
                        print(f"    Cardiac data loading failed for insula model: {fallback_error}")
                        continue
                else:
                    print(f"    Baseline model - no cardiac data needed")

                # Run the model
                trial, train_stepper = runner.run(**condition_with_cardiac)

                # Extract results
                outputs = train_stepper.outputs.detach().cpu().numpy()
                states = [state.detach().cpu().numpy() for state in train_stepper.state_collector]
                cost = train_stepper.cost.item()

                # Check for NaN values
                has_nan_outputs = np.isnan(outputs).any()
                has_nan_states = any(np.isnan(state).any() for state in states)
                has_nan_cost = np.isnan(cost)

                # Calculate performance
                performance = self._calculate_interval_comparison_performance(trial, outputs)

                # Only mark as has_nan if the accuracy calculation itself failed
                performance_has_critical_nan = False
                if 'accuracy' in performance:
                    if np.isnan(performance['accuracy']):
                        performance_has_critical_nan = True

                # Only mark as has_nan for truly critical failures
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
                    'performance': {'accuracy': np.nan},
                    'cost': np.nan,
                    'has_nan': True,
                    'cardiac_library': cardiac_library,
                    'cardiac_metadata': {'error': str(e)}
                }
                results['test_results'].append(test_result)
                continue

        # Summarize performance using the same logic as the working tester
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance_fixed(results['test_results'])

        return results

    def _summarize_performance_fixed(self, test_results):
        """Summarize performance using the same logic as the working tester"""

        if not test_results:
            return {}

        # Use the same filtering logic as interval_comparison_tester.py
        # Don't filter by has_nan - only filter by actual NaN values in the metrics
        accuracies = [r['performance']['accuracy'] for r in test_results
                      if 'accuracy' in r['performance'] and not np.isnan(r['performance']['accuracy'])]

        costs = [r['cost'] for r in test_results if not np.isnan(r['cost']) and np.isfinite(r['cost'])]

        # Calculate summary with better error handling
        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(accuracies),
            'nan_rate': (len(test_results) - len(accuracies)) / len(test_results) if test_results else 0,
            'mean_accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'mean_cost': np.mean(costs) if costs else np.inf
        }

        # Debug output to see what's happening
        print(f"    Summary: {len(accuracies)} valid accuracies out of {len(test_results)} conditions")
        if accuracies:
            print(f"    Mean accuracy: {summary['mean_accuracy']:.4f}")
        else:
            print(f"    No valid accuracies found!")

        return summary

    def _generate_written_report(self, all_results):
        """Generate comprehensive written report for cardiac library comparison"""

        report_path = os.path.join(self.base_results_dir, "insula_cardiac_library_comparison_report.txt")

        # Prepare data for analysis
        performance_data = []
        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    performance_data.append({
                        'model': model_name,
                        'library': lib_name,
                        'mean_accuracy': lib_results['performance_summary']['mean_accuracy'],
                        'std_accuracy': lib_results['performance_summary']['std_accuracy'],
                        'mean_cost': lib_results['performance_summary']['mean_cost'],
                        'nan_rate': lib_results['performance_summary']['nan_rate'],
                        'num_conditions': lib_results['performance_summary']['num_conditions'],
                        'num_valid': lib_results['performance_summary']['num_valid']
                    })

        df = pd.DataFrame(performance_data)

        with open(report_path, 'w') as f:
            f.write("INTERVAL COMPARISON INSULA CARDIAC LIBRARY COMPARISON REPORT\n")
            f.write("=" * 85 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Insula Models Tested: {len(all_results)}\n")
            f.write(f"Cardiac Libraries: {len(self.cardiac_libraries)}\n")
            f.write(f"Total Comparisons: {len(performance_data)}\n")
            f.write(
                f"Overall Mean Accuracy: {df['mean_accuracy'].mean():.4f} ± {df['mean_accuracy'].std():.4f}\n")
            f.write(f"Accuracy Range: {df['mean_accuracy'].min():.4f} - {df['mean_accuracy'].max():.4f}\n")

            # Best and worst performing combinations
            best_combo = df.loc[df['mean_accuracy'].idxmax()]
            worst_combo = df.loc[df['mean_accuracy'].idxmin()]
            f.write(
                f"Best Combination: {best_combo['model']} + {best_combo['library']} ({best_combo['mean_accuracy']:.4f})\n")
            f.write(
                f"Worst Combination: {worst_combo['model']} + {worst_combo['library']} ({worst_combo['mean_accuracy']:.4f})\n\n")

            # Library Rankings
            f.write("CARDIAC LIBRARY PERFORMANCE RANKINGS\n")
            f.write("-" * 45 + "\n")

            library_means = df.groupby('library')['mean_accuracy'].mean().sort_values(ascending=False)

            f.write("Ranked by Mean Accuracy (Higher = Better):\n\n")
            for rank, (lib_name, mean_accuracy) in enumerate(library_means.items(), 1):
                lib_data = df[df['library'] == lib_name]
                std_accuracy = lib_data['mean_accuracy'].std()
                count = len(lib_data)
                mean_cost = lib_data['mean_cost'].mean()
                nan_rate = lib_data['nan_rate'].mean()

                f.write(f"{rank:2d}. {lib_name:15s} | Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} "
                        f"| Cost: {mean_cost:.3f} | NaN Rate: {nan_rate:.1%} | n={count}\n")

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
            f.write("INSULA MODEL CONSISTENCY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            model_stats = df.groupby('model').agg({
                'mean_accuracy': ['mean', 'std'],
                'library': 'count'
            }).round(4)

            f.write("Insula models ranked by consistency (Lower Std = More Consistent across libraries):\n\n")
            model_consistency = df.groupby('model')['mean_accuracy'].std().sort_values()

            for rank, (model_name, std_accuracy) in enumerate(model_consistency.items(), 1):
                model_data = df[df['model'] == model_name]
                mean_accuracy = model_data['mean_accuracy'].mean()
                count = len(model_data)
                min_perf = model_data['mean_accuracy'].min()
                max_perf = model_data['mean_accuracy'].max()

                f.write(f"{rank:2d}. {model_name:20s} | Mean: {mean_accuracy:.4f} | Std: {std_accuracy:.4f} "
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
                f.write(f"  Insula Models Tested: {len(lib_data)}\n")
                f.write(
                    f"  Mean Accuracy: {lib_data['mean_accuracy'].mean():.4f} ± {lib_data['mean_accuracy'].std():.4f}\n")
                f.write(
                    f"  Accuracy Range: [{lib_data['mean_accuracy'].min():.4f}, {lib_data['mean_accuracy'].max():.4f}]\n")
                f.write(f"  Mean Cost: {lib_data['mean_cost'].mean():.4f}\n")
                f.write(f"  Average NaN Rate: {lib_data['nan_rate'].mean():.1%}\n")

                # Best and worst models for this library
                best_model = lib_data.loc[lib_data['mean_accuracy'].idxmax()]
                worst_model = lib_data.loc[lib_data['mean_accuracy'].idxmin()]
                f.write(f"  Best Insula Model: {best_model['model']} ({best_model['mean_accuracy']:.4f})\n")
                f.write(f"  Worst Insula Model: {worst_model['model']} ({worst_model['mean_accuracy']:.4f})\n")

            # Detailed Model Analysis
            f.write("\n\nDETAILED INSULA MODEL ANALYSIS\n")
            f.write("-" * 35 + "\n")

            for model_name in sorted(df['model'].unique()):
                model_data = df[df['model'] == model_name]

                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Libraries Tested: {len(model_data)}\n")
                f.write(
                    f"  Mean Accuracy: {model_data['mean_accuracy'].mean():.4f} ± {model_data['mean_accuracy'].std():.4f}\n")
                f.write(
                    f"  Accuracy Range: [{model_data['mean_accuracy'].min():.4f}, {model_data['mean_accuracy'].max():.4f}]\n")
                f.write(f"  Mean Cost: {model_data['mean_cost'].mean():.4f}\n")
                f.write(f"  Average NaN Rate: {model_data['nan_rate'].mean():.1%}\n")

                # Best and worst libraries for this model
                best_library = model_data.loc[model_data['mean_accuracy'].idxmax()]
                worst_library = model_data.loc[model_data['mean_accuracy'].idxmin()]
                f.write(f"  Best Library: {best_library['library']} ({best_library['mean_accuracy']:.4f})\n")
                f.write(f"  Worst Library: {worst_library['library']} ({worst_library['mean_accuracy']:.4f})\n")
                f.write(f"  Library Sensitivity: {model_data['mean_accuracy'].std():.4f} (lower = more robust)\n")

            # Statistical Analysis
            f.write("\n\nSTATISTICAL ANALYSIS\n")
            f.write("-" * 25 + "\n")

            # ANOVA-like analysis
            total_variance = df['mean_accuracy'].var()
            between_library_variance = df.groupby('library')['mean_accuracy'].mean().var()
            between_model_variance = df.groupby('model')['mean_accuracy'].mean().var()

            f.write(f"Total Performance Variance: {total_variance:.6f}\n")
            f.write(
                f"Between-Library Variance: {between_library_variance:.6f} ({between_library_variance / total_variance:.1%} of total)\n")
            f.write(
                f"Between-Model Variance: {between_model_variance:.6f} ({between_model_variance / total_variance:.1%} of total)\n")

            # Correlation analysis
            f.write(f"\nPerformance Correlations:\n")
            f.write(f"  Accuracy vs Cost: {df['mean_accuracy'].corr(df['mean_cost']):.3f}\n")
            f.write(f"  Accuracy vs NaN Rate: {df['mean_accuracy'].corr(df['nan_rate']):.3f}\n")

            # Top and bottom performing combinations
            f.write(f"\n\nTOP 10 INSULA MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 45 + "\n")
            top_combinations = df.nlargest(10, 'mean_accuracy')
            for i, row in top_combinations.iterrows():
                f.write(
                    f"{row.name + 1:2d}. {row['model']:20s} + {row['library']:15s} | {row['mean_accuracy']:.4f}\n")

            f.write(f"\n\nBOTTOM 10 INSULA MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 47 + "\n")
            bottom_combinations = df.nsmallest(10, 'mean_accuracy')
            for i, row in bottom_combinations.iterrows():
                f.write(
                    f"{len(df) - row.name:2d}. {row['model']:20s} + {row['library']:15s} | {row['mean_accuracy']:.4f}\n")

            # Recommendations
            f.write(f"\n\nRECOMMENDATIONS FOR INSULA MODELS\n")
            f.write("-" * 40 + "\n")

            best_overall_library = library_means.index[0]
            most_consistent_library = df.groupby('library')['mean_accuracy'].std().idxmin()
            most_robust_model = model_consistency.index[0]
            best_overall_model = df.groupby('model')['mean_accuracy'].mean().idxmax()

            f.write(f"1. BEST OVERALL LIBRARY FOR INSULA: {best_overall_library}\n")
            f.write(f"   Highest mean performance across all insula models\n\n")

            f.write(f"2. MOST CONSISTENT LIBRARY FOR INSULA: {most_consistent_library}\n")
            f.write(f"   Lowest performance variance across insula models\n\n")

            f.write(f"3. MOST ROBUST INSULA MODEL: {most_robust_model}\n")
            f.write(f"   Lowest performance variance across libraries\n\n")

            f.write(f"4. BEST PERFORMING INSULA MODEL: {best_overall_model}\n")
            f.write(f"   Highest mean performance across all libraries\n\n")

            if library_cv > 0.1:
                f.write(f"5. LIBRARY SELECTION IS CRITICAL FOR INSULA:\n")
                f.write(f"   High coefficient of variation ({library_cv:.3f}) indicates\n")
                f.write(f"   cardiac library choice significantly impacts insula model performance.\n")
                f.write(f"   Recommend using {best_overall_library} for optimal results.\n\n")
            else:
                f.write(f"5. LIBRARY SELECTION IS NOT CRITICAL FOR INSULA:\n")
                f.write(f"   Low coefficient of variation ({library_cv:.3f}) indicates\n")
                f.write(f"   insula models are robust to cardiac library choice.\n\n")

            # Technical Notes
            f.write(f"TECHNICAL NOTES - INSULA MODELS\n")
            f.write("-" * 35 + "\n")
            f.write(f"- Accuracy: Proportion of correct interval comparisons\n")
            f.write(f"- NaN rate: Proportion of trials producing invalid outputs\n")
            f.write(f"- Library CV: Coefficient of variation of library means\n")
            f.write(f"- Model sensitivity: Standard deviation across libraries\n")
            f.write(f"- Insula processing: ECG → aINS activity → RNN modulation\n")
            f.write(
                f"- Total conditions tested per model-library pair: {df['num_conditions'].iloc[0] if len(df) > 0 else 'N/A'}\n")

            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")

        print(f"Comprehensive insula library comparison report saved to: {report_path}")
        return report_path

    def _calculate_interval_comparison_performance(self, trial, outputs):
        """Calculate interval comparison performance metrics"""

        batch_size = outputs.shape[1]

        # Get target choice: 1 if I1>I2, -1 if I2>=I1
        interval1 = trial.prod_interval1 * 20  # Convert to ms
        interval2 = trial.prod_interval2 * 20  # Convert to ms
        target_choice = 2 * (interval1 > interval2) - 1

        # Get final outputs (average of last 20 timesteps)
        final_outputs = outputs[-20:, :, :].mean(axis=0)  # [B, 2]

        # Convert to choice format: positive = choose I1, negative = choose I2
        network_choice = final_outputs[:, 0] - final_outputs[:, 1]

        # Calculate accuracy
        choice_correct = (network_choice * target_choice > 0)  # Both same sign = correct
        accuracy = choice_correct.mean()

        return {
            'accuracy': float(accuracy),
            'network_choices': network_choice.tolist(),
            'target_choices': [target_choice] * batch_size,  # Convert scalar to list
            'choice_correct': choice_correct.tolist(),
            'interval1_ms': interval1.tolist() if hasattr(interval1, 'tolist') else [interval1] * batch_size,
            'interval2_ms': interval2.tolist() if hasattr(interval2, 'tolist') else [interval2] * batch_size
        }

    def test_model_across_libraries(self, model_dir, test_conditions=None):
        """Test a single insula model across all available cardiac libraries"""

        if test_conditions is None:
            test_conditions = self.get_test_conditions()

        model_name = os.path.basename(model_dir)
        print(f"\nTesting insula model {model_name} across {len(self.cardiac_libraries)} cardiac libraries")

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
                    f"    Completed {library_name}: mean_accuracy = {library_result['performance_summary']['mean_accuracy']:.4f}")
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
        library_costs = {}

        for lib_name, lib_results in valid_libraries.items():
            summary = lib_results['performance_summary']
            library_performances[lib_name] = summary['mean_accuracy']
            library_costs[lib_name] = summary['mean_cost']

        # Calculate comparison statistics
        performances = list(library_performances.values())

        comparison = {
            'library_performances': library_performances,
            'library_costs': library_costs,
            'overall_mean_accuracy': np.mean(performances),
            'overall_std_accuracy': np.std(performances),
            'accuracy_range': max(performances) - min(performances),
            'best_library': max(library_performances.items(), key=lambda x: x[1])[0],
            'worst_library': min(library_performances.items(), key=lambda x: x[1])[0],
            'performance_coefficient_of_variation': np.std(performances) / np.mean(performances) if np.mean(
                performances) > 0 else np.inf
        }

        return comparison

    def test_multiple_models_across_libraries(self, model_directories, num_conditions=15, use_dataset=True,
                                              include_ood=True):
        """Test multiple insula models across all libraries"""

        # Get test conditions
        test_conditions = self.get_test_conditions(num_conditions, use_dataset, include_ood)

        # Filter for insula models only (not baseline or piezo models)
        valid_models = []
        for model_info in model_directories:
            if len(model_info) == 3:
                model_dir, rule_name, run_name = model_info
            else:
                model_dir, rule_name = model_info
                run_name = os.path.basename(model_dir)

            # Only include insula models since they're the only ones that use cardiac data
            if self._is_insula_model(model_dir):
                valid_models.append((model_dir, rule_name, run_name))

        print(f"Found {len(valid_models)} insula models to test")

        all_results = []

        for model_dir, rule_name, run_name in valid_models:
            print(f"\n{'=' * 60}")
            print(f"Testing insula model: {run_name}")
            print(f"{'=' * 60}")

            results = self.test_model_across_libraries(model_dir, test_conditions)

            if results and results['library_results']:
                results['model_name'] = run_name
                all_results.append(results)

                # Save individual results
                results_path = os.path.join(self.base_results_dir, f"{run_name}_insula_library_test.pkl")
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Saved results: {results_path}")

        if all_results:
            self._generate_cross_model_analysis(all_results)
            self._create_comparison_plots(all_results)
            self._generate_written_report(all_results)

            # Save consolidated results
            summary_path = os.path.join(self.base_results_dir, "all_insula_library_test_results.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

        return all_results

    def _generate_cross_model_analysis(self, all_results):
        """Generate analysis across all insula models and libraries"""

        print(f"\nGenerating cross-model analysis for insula models...")

        # Create performance matrix: models x libraries
        performance_data = []
        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    performance_data.append({
                        'model': model_name,
                        'library': lib_name,
                        'mean_accuracy': lib_results['performance_summary']['mean_accuracy'],
                        'mean_cost': lib_results['performance_summary']['mean_cost'],
                        'nan_rate': lib_results['performance_summary']['nan_rate']
                    })

        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data)

        # Generate summary statistics
        analysis = {
            'overall_stats': {
                'num_insula_models': len(all_results),
                'num_libraries': len(self.cardiac_libraries),
                'total_comparisons': len(performance_data),
                'mean_accuracy_overall': df['mean_accuracy'].mean(),
                'std_accuracy_overall': df['mean_accuracy'].std()
            },
            'library_rankings': df.groupby('library')['mean_accuracy'].agg(['mean', 'std', 'count']).to_dict(),
            'model_consistency': df.groupby('model')['mean_accuracy'].agg(['mean', 'std', 'count']).to_dict(),
            'library_effect_size': df.groupby('library')['mean_accuracy'].mean().std(),
            'best_combinations': df.nlargest(5, 'mean_accuracy')[['model', 'library', 'mean_accuracy']].to_dict(
                'records'),
            'worst_combinations': df.nsmallest(5, 'mean_accuracy')[['model', 'library', 'mean_accuracy']].to_dict(
                'records')
        }

        # Save analysis
        analysis_path = os.path.join(self.base_results_dir, "cross_insula_model_library_analysis.json")
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2)

        print(f"Cross-insula-model analysis saved to: {analysis_path}")

        # Print summary
        print(f"\nLIBRARY RANKINGS FOR INSULA MODELS (by mean accuracy):")
        library_means = df.groupby('library')['mean_accuracy'].mean().sort_values(ascending=False)
        for lib, perf in library_means.items():
            print(f"  {lib}: {perf:.4f}")

        print(f"\nINSULA MODEL CONSISTENCY (by accuracy std - lower is more consistent):")
        model_stds = df.groupby('model')['mean_accuracy'].std().sort_values()
        for model, std in model_stds.items():
            print(f"  {model}: {std:.4f}")

    def _create_comparison_plots(self, all_results):
        """Create visualization plots for the insula library comparison"""

        # Prepare data for plotting
        models = []
        libraries = []
        accuracies = []

        for result in all_results:
            model_name = result['model_name']
            for lib_name, lib_results in result['library_results'].items():
                if lib_results and 'performance_summary' in lib_results:
                    models.append(model_name)
                    libraries.append(lib_name)
                    accuracies.append(lib_results['performance_summary']['mean_accuracy'])

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Model': models,
            'Library': libraries,
            'Accuracy': accuracies
        })

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Interval Comparison: Insula Model Cardiac Library Comparison', fontsize=16)

        # Plot 1: Library performance boxplot
        ax1 = axes[0, 0]
        libraries_unique = df['Library'].unique()
        library_data = [df[df['Library'] == lib]['Accuracy'].values for lib in libraries_unique]
        ax1.boxplot(library_data, labels=libraries_unique)
        ax1.set_title('Performance Distribution by Library\n(Insula Models)')
        ax1.set_ylabel('Mean Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Model consistency
        ax2 = axes[0, 1]
        model_means = df.groupby('Model')['Accuracy'].mean()
        model_stds = df.groupby('Model')['Accuracy'].std()
        ax2.scatter(model_means, model_stds, alpha=0.7, s=100, color='green')
        ax2.set_xlabel('Mean Accuracy')
        ax2.set_ylabel('Std Accuracy')
        ax2.set_title('Insula Model Consistency\n(lower std = more consistent)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Heatmap of model x library performance
        ax3 = axes[1, 0]
        pivot_data = df.pivot(index='Model', columns='Library', values='Accuracy')
        im = ax3.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels(pivot_data.columns, rotation=45)
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels(pivot_data.index)
        ax3.set_title('Insula Model × Library Performance Heatmap')
        plt.colorbar(im, ax=ax3, label='Mean Accuracy')

        # Plot 4: Library effect size
        ax4 = axes[1, 1]
        library_means = df.groupby('Library')['Accuracy'].mean().sort_values(ascending=False)
        library_stds = df.groupby('Library')['Accuracy'].std().reindex(library_means.index)
        x_pos = range(len(library_means))
        ax4.bar(x_pos, library_means.values, yerr=library_stds.values,
                capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(library_means.index, rotation=45)
        ax4.set_ylabel('Mean Accuracy')
        ax4.set_title('Library Performance Ranking\n(Insula Models)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, "insula_cardiac_library_comparison_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Insula library comparison plots saved to: {plot_path}")

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

    def discover_insula_models(self):
        """Discover trained interval comparison insula models"""
        model_dirs = []

        possible_base_dirs = [
            'enhanced_insula_comparison_results',
            'insula_comparison_results',
            'enhanced_piezo_comparison_results',  # Keep for backwards compatibility
            'comparison_results',
            'interval_comparison_results',
            'model/interval_comparison',
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                print(f"Searching in {base_dir}...")
                self._search_model_subdirs(base_dir, 'interval_comparison', model_dirs)

        return model_dirs

    def _search_model_subdirs(self, base_dir, rule_name, model_dirs):
        """Search for insula and baseline model subdirectories, ignoring piezo models"""
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Only include insula models and baseline (no_piezo) models
                    # Ignore pure piezo models (piezo_run_* without insula)
                    if ('insula_run_' in item or 'no_piezo_run_' in item or 
                        'no_insula' in item or 'baseline_run_' in item):
                        # Check if this run directory has valid checkpoints
                        best_model_path = self._find_best_model_checkpoint(item_path)
                        if best_model_path:
                            # Verify this is actually an insula model by checking hp.json
                            if self._is_insula_model(best_model_path):
                                # Store the BASE model directory, not the specific checkpoint
                                model_dirs.append((item_path, rule_name, item))
                                print(f"  Found model: {item}")
                            else:
                                print(f"  Skipped non-insula model: {item}")
                    elif self._has_valid_model(item_path):
                        # This directory itself contains a model - check if it's insula
                        if self._is_insula_model(item_path):
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

    def _is_insula_model(self, model_path):
        """Check if a model is an insula model (only insula models use cardiac data)"""
        hp_path = os.path.join(model_path, 'hp.json')
        try:
            with open(hp_path, 'r') as f:
                hp = json.load(f)
            
            use_insula = hp.get('use_insula', False)
            
            # Only include insula models - baseline models don't use cardiac data
            return use_insula
                
        except Exception as e:
            print(f"  Warning: Could not read hp.json from {model_path}: {e}")
            return False

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
                    
                    # Look for appropriate metric for interval comparison tasks
                    timing_error = None
                    if 'mean_choice_error' in checkpoint_log:
                        # For comparison tasks, use choice error (lower is better)
                        timing_error = checkpoint_log['mean_choice_error']
                    elif 'cost' in checkpoint_log:
                        # Fallback to cost (lower is better)
                        timing_error = checkpoint_log['cost']
                    elif 'mean_rel_action_time' in checkpoint_log:
                        # For production tasks
                        timing_error = checkpoint_log['mean_rel_action_time']
                    
                    if timing_error is not None and np.isfinite(timing_error) and timing_error < min_timing_error:
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
    """Run the insula cardiac library comparison test"""

    # Initialize the tester
    tester = IntervalComparisonInsulaLibraryTester()

    # Discover interval comparison insula models
    model_directories = tester.discover_insula_models()

    if not model_directories:
        print("No interval comparison insula models found!")
        return

    print(f"Found {len(model_directories)} interval comparison insula models")

    # Run the comparison across libraries
    results = tester.test_multiple_models_across_libraries(model_directories)

    print(f"\nInsula cardiac library comparison complete!")
    print(f"Results saved to: {tester.base_results_dir}")

    return results


if __name__ == "__main__":
    main()
