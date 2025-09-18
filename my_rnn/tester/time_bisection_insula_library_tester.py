"""
Time Bisection Insula Library Comparison Tester
===============================================

Tests time bisection insula models across different cardiac libraries
to assess robustness and optimal cardiac data sources.

Based on time_bisection_library_tester.py but adapted for insula models.
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


class TimeBisectionInsulaLibraryTester:
    """Test time bisection insula models across different cardiac libraries"""

    def __init__(self, base_results_dir="time_bisection_insula_cardiac_library_results"):
        self.base_results_dir = base_results_dir
        self.rule_name = 'time_bisection'

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

        print(f"Time Bisection Insula Library Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Available cardiac libraries: {list(self.cardiac_libraries.keys())}")

    def get_test_conditions(self, num_conditions=10, use_dataset=True, include_ood=True):
        """Get test conditions for time bisection - dataset sampling or predefined"""

        if use_dataset:
            # Load from your existing dataset (consistent with other testers)
            dataset_path = "enhanced_interval_datasets/time_bisection_dataset.json"

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
                        {'duration': 2200, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True, 'ood_type': 'above_range'},
                        {'duration': 2400, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True, 'ood_type': 'above_range'},
                        {'duration': 2600, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True, 'ood_type': 'above_range'},
                        {'duration': 2800, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True, 'ood_type': 'above_range'},
                        {'duration': 400, 'short_standard': 300, 'long_standard': 900, 'is_ood': True, 'ood_type': 'below_range'},
                        {'duration': 600, 'short_standard': 300, 'long_standard': 900, 'is_ood': True, 'ood_type': 'below_range'},
                        {'duration': 800, 'short_standard': 300, 'long_standard': 900, 'is_ood': True, 'ood_type': 'below_range'},
                    ]
                    selected.extend(ood_conditions)

                print(
                    f"Using dataset conditions: {len(selected)} total ({len(selected) - (len(ood_conditions) if include_ood else 0)} from dataset)")
                return selected

            else:
                print(f"Dataset not found at {dataset_path}, falling back to predefined conditions")

        # Fallback to predefined conditions (original behavior)
        conditions = [
            {'duration': 1000, 'short_standard': 300, 'long_standard': 900},
            {'duration': 1200, 'short_standard': 300, 'long_standard': 900},
            {'duration': 1500, 'short_standard': 300, 'long_standard': 900},
            {'duration': 1800, 'short_standard': 300, 'long_standard': 900},
            {'duration': 2000, 'short_standard': 300, 'long_standard': 900},
        ]

        # Add OOD conditions if requested
        if include_ood:
            conditions.extend([
                {'duration': 2200, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'duration': 2400, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'duration': 2600, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'duration': 2800, 'short_standard': 2000, 'long_standard': 3000, 'is_ood': True,
                 'ood_type': 'above_range'},
                {'duration': 400, 'short_standard': 300, 'long_standard': 900, 'is_ood': True,
                 'ood_type': 'below_range'},
                {'duration': 600, 'short_standard': 300, 'long_standard': 900, 'is_ood': True,
                 'ood_type': 'below_range'},
                {'duration': 800, 'short_standard': 300, 'long_standard': 900, 'is_ood': True,
                 'ood_type': 'below_range'},
            ])

        print(f"Using predefined conditions: {len(conditions)} total")
        return conditions

    def test_model_with_single_library(self, model_dir, cardiac_library, test_conditions):
        """Test a single model with one specific cardiac library - FIXED AND DEBUGGED VERSION"""

        # Verify model exists and has insula
        analysis = tools.analyze_model_directory(model_dir)
        if not analysis['exists']:
            print(f"Model directory not found: {model_dir}")
            return None

        if not analysis['use_insula']:
            print(f"Model {model_dir} is not an insula model - skipping")
            return None

        model_name = os.path.basename(model_dir)
        print(f"  Testing {model_name} with {cardiac_library}")

        # Initialize runner with fixed hyperparameters
        hp = tools.load_hp(model_dir)
        if 'rnn_type' not in hp:
            hp['rnn_type'] = 'RNN'
        if 'alpha' not in hp:
            hp['alpha'] = 1.0 * hp.get('dt', 20) / hp.get('tau', 20)

        runner = Runner(
            rule_name=self.rule_name,
            model_dir=model_dir,
            hp=hp,
            is_cuda=False,
            noise_on=False
        )

        results = {
            'model_dir': model_dir,
            'model_name': model_name,
            'cardiac_library': cardiac_library,
            'test_results': [],
            'performance_summary': {},
            'debug_info': []  # ADD DEBUG INFO
        }

        # Test each condition with the specified cardiac library
        for i, condition in enumerate(test_conditions):
            print(f"    Condition {i + 1}/{len(test_conditions)}: duration={condition['duration']}ms")

            debug_info = {
                'condition_idx': i,
                'cardiac_library': cardiac_library,
                'condition': condition
            }

            try:
                # FIXED: Direct cardiac data loading instead of monkey-patching
                task_duration_ms = condition['duration'] + 500  # stimulus + delay + response

                # Generate cardiac data directly with specified library
                cardiac_data = create_real_cardiac_data_for_task(
                    task_duration_ms=task_duration_ms,
                    dt=runner.hp.get('dt', 20),
                    slice_size=runner.hp.get('heartbeat_slice_size', 20),
                    cardiac_library=cardiac_library  # Direct parameter passing
                )

                # DEBUG: Verify cardiac data is different between libraries
                debug_info['cardiac_heart_rate'] = cardiac_data['heart_rate']
                debug_info['cardiac_r_peaks'] = len(cardiac_data['r_peak_times'])
                debug_info['cardiac_pressure_mean'] = cardiac_data['pressure_mean']
                debug_info['cardiac_pressure_std'] = cardiac_data['pressure_std']

                # Convert to tensor
                hb_sequence = torch.tensor(cardiac_data['hb_sequence'], dtype=torch.float32)
                condition_with_cardiac = condition.copy()
                condition_with_cardiac['hb_sequence'] = hb_sequence

                # DEBUG: Print first few values to verify uniqueness
                if i == 0:  # Only for first condition
                    pressure_sample = cardiac_data['hb_sequence'][:5, 1]  # First 5 pressure values
                    print(f"    DEBUG: Cardiac pressure sample for {cardiac_library}: {pressure_sample}")

                # Run the model
                trial, train_stepper = runner.run(**condition_with_cardiac)

                # DEBUG: Check model outputs
                outputs = train_stepper.outputs.detach().cpu().numpy()
                debug_info['output_shape'] = outputs.shape
                debug_info['output_mean'] = outputs.mean()
                debug_info['output_std'] = outputs.std()
                debug_info['output_range'] = [outputs.min(), outputs.max()]

                # DEBUG: Check targets
                targets = trial.y
                debug_info['target_shape'] = targets.shape
                debug_info['target_sum'] = targets.sum()
                debug_info['target_nonzero'] = (targets != 0).sum()

                # Calculate performance with debugging
                performance = self._calculate_time_bisection_performance(
                    trial, outputs, condition
                )

                # Check for critical failures
                has_nan_outputs = np.isnan(outputs).any()
                has_nan_cost = np.isnan(train_stepper.cost.item())
                performance_has_critical_nan = np.isnan(performance.get('accuracy', 0))

                has_nan = has_nan_outputs or has_nan_cost or performance_has_critical_nan

                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'performance': performance,
                    'cost': train_stepper.cost.item(),
                    'has_nan': has_nan,
                    'cardiac_library': cardiac_library,
                    'debug_info': debug_info
                }

                results['test_results'].append(test_result)

                # Print debug info for first few conditions
                if i < 3:
                    print(f"      DEBUG: Accuracy = {performance['accuracy']:.4f}")
                    print(f"      DEBUG: Cardiac HR = {debug_info['cardiac_heart_rate']:.1f} BPM")
                    print(f"      DEBUG: Output mean = {debug_info['output_mean']:.4f}")

            except Exception as e:
                print(f"    ERROR in condition {i + 1}: {e}")
                import traceback
                traceback.print_exc()

                # Add failed result
                test_result = {
                    'condition_idx': i,
                    'condition': {k: v for k, v in condition.items() if k != 'hb_sequence'},
                    'performance': {'accuracy': np.nan},
                    'cost': np.nan,
                    'has_nan': True,
                    'cardiac_library': cardiac_library,
                    'debug_info': {'error': str(e)}
                }
                results['test_results'].append(test_result)
                continue

        # Calculate summary with debugging
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance_with_debug(results['test_results'])

            # Add debug summary
            cardiac_hrs = [r['debug_info'].get('cardiac_heart_rate', 0) for r in results['test_results']
                           if 'debug_info' in r and 'cardiac_heart_rate' in r['debug_info']]
            results['debug_summary'] = {
                'cardiac_hr_range': [min(cardiac_hrs), max(cardiac_hrs)] if cardiac_hrs else [0, 0],
                'cardiac_hr_mean': np.mean(cardiac_hrs) if cardiac_hrs else 0,
                'num_valid_cardiac': len(cardiac_hrs)
            }

            print(f"    DEBUG SUMMARY for {cardiac_library}:")
            print(f"      Mean accuracy: {results['performance_summary']['accuracy']:.4f}")
            print(f"      Cardiac HR range: {results['debug_summary']['cardiac_hr_range']}")
            print(f"      Valid cardiac data: {results['debug_summary']['num_valid_cardiac']}")

        return results

    def _calculate_time_bisection_performance(self, trial, outputs, condition):
        """Calculate time bisection performance with target mismatch debugging"""

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

        # DEBUG: Check for target generation mismatch
        duration = condition.get('duration', 0)
        short_std = condition.get('short_standard', 1000)
        long_std = condition.get('long_standard', 2000)

        # What SHOULD the target be based on different logics?
        # Dataset logic (closest standard)
        short_distance = abs(duration - short_std)
        long_distance = abs(duration - long_std)
        expected_dataset_target = 0 if short_distance < long_distance else 1

        # Bisection point logic
        bisection_point = (short_std + long_std) / 2
        expected_bisection_target = 0 if duration <= bisection_point else 1

        # What did the task actually generate?
        actual_target = target_classes[0] if len(target_classes) > 0 else -1

        # Check for mismatch
        dataset_mismatch = actual_target != expected_dataset_target
        bisection_mismatch = actual_target != expected_bisection_target

        # Print debug info for mismatches or first few conditions
        if dataset_mismatch or bisection_mismatch:
            print(f"\n*** TARGET MISMATCH DETECTED ***")
            print(f"Duration: {duration}ms, Standards: {short_std}ms/{long_std}ms")
            print(f"Expected (dataset/closest): {expected_dataset_target}")
            print(f"Expected (bisection): {expected_bisection_target}")
            print(f"Actual target: {actual_target}")
            print(f"Model prediction: {pred_classes[0] if len(pred_classes) > 0 else 'none'}")
            print(f"Bisection point: {bisection_point}ms")

        # Calculate accuracy
        correct = (pred_classes == target_classes).astype(float)
        accuracy = float(np.mean(correct)) if correct.size > 0 else 0.0

        return {
            'accuracy': accuracy,
            'duration': duration,
            'actual_target': int(actual_target),
            'expected_dataset_target': expected_dataset_target,
            'expected_bisection_target': expected_bisection_target,
            'target_mismatch': dataset_mismatch or bisection_mismatch,
            'model_prediction': int(pred_classes[0]) if len(pred_classes) > 0 else -1
        }

    def _summarize_performance_with_debug(self, test_results):
        """Summarize performance with debugging information AND ID/OOD breakdown"""

        if not test_results:
            return {}

        # Extract accuracies and debug info
        accuracies = []
        id_accuracies = []  # ADD: Track ID accuracies
        ood_accuracies = []  # ADD: Track OOD accuracies

        debug_summary = {
            'target_mismatches': 0,
            'total_conditions': len(test_results),
            'accuracy_distribution': [],
            'pred_class_distribution': {'class_0': 0, 'class_1': 0},
            'target_class_distribution': {'class_0': 0, 'class_1': 0}
        }

        for r in test_results:
            perf = r.get('performance', {})
            debug = r.get('debug_info', {})
            condition = r.get('condition', {})  # ADD: Get condition info

            if 'accuracy' in perf and not np.isnan(perf['accuracy']):
                accuracy = perf['accuracy']
                accuracies.append(accuracy)
                debug_summary['accuracy_distribution'].append(accuracy)

                # ADD: Separate ID and OOD accuracies
                is_ood = condition.get('is_ood', False)
                if is_ood:
                    ood_accuracies.append(accuracy)
                else:
                    id_accuracies.append(accuracy)

                # Check for target mismatches
                if perf.get('target_mismatch', False):
                    debug_summary['target_mismatches'] += 1

        costs = [r['cost'] for r in test_results if not np.isnan(r['cost']) and np.isfinite(r['cost'])]

        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(accuracies),
            'nan_rate': (len(test_results) - len(accuracies)) / len(test_results) if test_results else 0,
            'accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'mean_cost': np.mean(costs) if costs else np.inf,
            'debug_summary': debug_summary,
            # ADD: ID/OOD metrics
            'mean_id_accuracy': np.mean(id_accuracies) if id_accuracies else np.nan,
            'mean_ood_accuracy': np.mean(ood_accuracies) if ood_accuracies else np.nan,
            'num_id_conditions': len(id_accuracies),
            'num_ood_conditions': len(ood_accuracies),
            'std_id_accuracy': np.std(id_accuracies) if id_accuracies else np.nan,
            'std_ood_accuracy': np.std(ood_accuracies) if ood_accuracies else np.nan,
        }

        # Print debug summary
        print(f"    DEBUG PERFORMANCE SUMMARY:")
        print(f"      Valid accuracies: {len(accuracies)}/{len(test_results)}")
        print(f"      Mean accuracy: {summary['accuracy']:.4f}")
        print(f"      ID conditions: {len(id_accuracies)}, OOD conditions: {len(ood_accuracies)}")
        if id_accuracies:
            print(f"      Mean ID accuracy: {summary['mean_id_accuracy']:.4f}")
        if ood_accuracies:
            print(f"      Mean OOD accuracy: {summary['mean_ood_accuracy']:.4f}")
        print(f"      Target mismatches: {debug_summary['target_mismatches']}/{debug_summary['total_conditions']}")

        if debug_summary['target_mismatches'] > 0:
            print(f"      WARNING: {debug_summary['target_mismatches']} target mismatches detected!")
            print(f"      This suggests target generation is inconsistent with dataset expectations")

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
                    f"    Completed {library_name}: accuracy = {library_result['performance_summary']['accuracy']:.4f}")
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
            library_performances[lib_name] = summary['accuracy']
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
        """Test multiple models across all libraries"""

        # Get test conditions
        test_conditions = self.get_test_conditions(num_conditions, use_dataset, include_ood)

        # Filter for insula models only (not baseline models)
        insula_models = []
        for model_info in model_directories:
            if len(model_info) == 3:
                model_dir, rule_name, run_name = model_info
            else:
                model_dir, rule_name = model_info
                run_name = os.path.basename(model_dir)

            analysis = tools.analyze_model_directory(model_dir)
            if analysis['use_insula']:
                insula_models.append((model_dir, rule_name, run_name))

        print(f"Found {len(insula_models)} insula models to test")

        all_results = []

        for model_dir, rule_name, run_name in insula_models:
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

            # Generate comprehensive written report
            self._generate_written_report(all_results)

            # Save consolidated results
            summary_path = os.path.join(self.base_results_dir, "all_cardiac_library_test_results.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

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
                        'accuracy': lib_results['performance_summary']['accuracy'],
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
                'mean_accuracy_overall': df['accuracy'].mean(),
                'std_accuracy_overall': df['accuracy'].std()
            },
            'library_rankings': df.groupby('library')['accuracy'].agg(['mean', 'std', 'count']).to_dict(),
            'model_consistency': df.groupby('model')['accuracy'].agg(['mean', 'std', 'count']).to_dict(),
            'library_effect_size': df.groupby('library')['accuracy'].mean().std(),
            'best_combinations': df.nlargest(5, 'accuracy')[['model', 'library', 'accuracy']].to_dict(
                'records'),
            'worst_combinations': df.nsmallest(5, 'accuracy')[['model', 'library', 'accuracy']].to_dict(
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
        print(f"\nLIBRARY RANKINGS (by mean accuracy):")
        library_means = df.groupby('library')['accuracy'].mean().sort_values(ascending=False)
        for lib, perf in library_means.items():
            print(f"  {lib}: {perf:.4f}")

        print(f"\nMODEL CONSISTENCY (by accuracy std - lower is more consistent):")
        model_stds = df.groupby('model')['accuracy'].std().sort_values()
        for model, std in model_stds.items():
            print(f"  {model}: {std:.4f}")

    def _create_comparison_plots(self, all_results):
        """Create visualization plots for the comparison"""

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
                    accuracies.append(lib_results['performance_summary']['accuracy'])

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Model': models,
            'Library': libraries,
            'Accuracy': accuracies
        })

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time Bisection Insula Models: Cardiac Library Comparison', fontsize=16)

        # Plot 1: Library performance boxplot
        ax1 = axes[0, 0]
        libraries_unique = df['Library'].unique()
        library_data = [df[df['Library'] == lib]['Accuracy'].values for lib in libraries_unique]
        ax1.boxplot(library_data, labels=libraries_unique)
        ax1.set_title('Performance Distribution by Library')
        ax1.set_ylabel('Mean Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')

        # Plot 2: Model consistency
        ax2 = axes[0, 1]
        model_means = df.groupby('Model')['Accuracy'].mean()
        model_stds = df.groupby('Model')['Accuracy'].std()
        ax2.scatter(model_means, model_stds, alpha=0.7, s=100)
        ax2.set_xlabel('Mean Accuracy')
        ax2.set_ylabel('Std Accuracy')
        ax2.set_title('Model Consistency (lower std = more consistent)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Heatmap of model x library performance
        ax3 = axes[1, 0]
        pivot_data = df.pivot(index='Model', columns='Library', values='Accuracy')
        im = ax3.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels(pivot_data.columns, rotation=45)
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels(pivot_data.index)
        ax3.set_title('Model × Library Performance Heatmap')
        plt.colorbar(im, ax=ax3, label='Mean Accuracy')

        # Plot 4: Library effect size
        ax4 = axes[1, 1]
        library_means = df.groupby('Library')['Accuracy'].mean().sort_values(ascending=False)
        library_stds = df.groupby('Library')['Accuracy'].std().reindex(library_means.index)
        x_pos = range(len(library_means))
        ax4.bar(x_pos, library_means.values, yerr=library_stds.values,
                capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(library_means.index, rotation=45)
        ax4.set_ylabel('Mean Accuracy')
        ax4.set_title('Library Performance Ranking')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance Level')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.base_results_dir, "cardiac_library_comparison_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plots saved to: {plot_path}")

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
                        'accuracy': lib_results['performance_summary']['accuracy'],
                        'std_accuracy': lib_results['performance_summary']['std_accuracy'],
                        'mean_cost': lib_results['performance_summary']['mean_cost'],
                        'nan_rate': lib_results['performance_summary']['nan_rate'],
                        'num_conditions': lib_results['performance_summary']['num_conditions'],
                        'num_valid': lib_results['performance_summary']['num_valid'],
                        'mean_id_accuracy': lib_results['performance_summary'].get('mean_id_accuracy', float('nan')),
                        'mean_ood_accuracy': lib_results['performance_summary'].get('mean_ood_accuracy', float('nan')),
                    })

        df = pd.DataFrame(performance_data)

        with open(report_path, 'w') as f:
            f.write("TIME BISECTION INSULA MODELS CARDIAC LIBRARY COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Insula Models Tested: {len(all_results)}\n")
            f.write(f"Cardiac Libraries: {len(self.cardiac_libraries)}\n")
            f.write(f"Total Comparisons: {len(performance_data)}\n")
            f.write(
                f"Overall Mean Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}\n")
            f.write(f"Performance Range: {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f}\n")

            # Best and worst performing combinations
            best_combo = df.loc[df['accuracy'].idxmax()]
            worst_combo = df.loc[df['accuracy'].idxmin()]
            f.write(
                f"Best Combination: {best_combo['model']} + {best_combo['library']} ({best_combo['accuracy']:.4f})\n")
            f.write(
                f"Worst Combination: {worst_combo['model']} + {worst_combo['library']} ({worst_combo['accuracy']:.4f})\n\n")

            # Library Rankings
            f.write("CARDIAC LIBRARY PERFORMANCE RANKINGS\n")
            f.write("-" * 45 + "\n")
            library_means = df.groupby('library')['accuracy'].mean().sort_values(ascending=False)

            f.write("Ranked by Mean Accuracy (Higher = Better):\n\n")
            for rank, (lib_name, mean_accuracy) in enumerate(library_means.items(), 1):
                lib_data = df[df['library'] == lib_name]
                std_accuracy = lib_data['accuracy'].std()
                count = len(lib_data)
                mean_cost = lib_data['mean_cost'].mean()
                nan_rate = lib_data['nan_rate'].mean()

                f.write(f"{rank:2d}. {lib_name:15s} | Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} "
                        f"| Cost: {mean_cost:.4f} | NaN Rate: {nan_rate:.1%} | n={count}\n")

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

            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")

        print(f"Comprehensive written report saved to: {report_path}")
        return report_path

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

    def discover_insula_bisection_models(self):
        """Discover trained time bisection insula models"""
        model_dirs = []

        possible_base_dirs = [
            'enhanced_piezo_time_bisection_results',  # Main directory for insula bisection models
            'enhanced_insula_time_bisection_results',  # Alternative naming
            'insula_time_bisection_results',
            'time_bisection_insula_results',
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
                    if ('insula_run_' in item or 'no_piezo_run_' in item or 
                        'no_insula' in item or 'baseline_run_' in item):
                        # Store the base directory, not the specific checkpoint
                        # Verify this is actually an insula model by checking hp.json
                        if self._is_insula_model(item_path):
                            model_dirs.append((item_path, rule_name, item))
                            print(f"  Found model: {item}")
                        else:
                            print(f"  Skipped non-insula model: {item}")
        except PermissionError:
            print(f"Permission denied accessing {base_dir}")
        except Exception as e:
            print(f"Error searching {base_dir}: {e}")

    def _is_insula_model(self, model_path):
        """Check if a model is an insula model (not baseline)"""
        hp_path = os.path.join(model_path, 'hp.json')
        try:
            with open(hp_path, 'r') as f:
                hp = json.load(f)
            
            use_insula = hp.get('use_insula', False)
            
            # Only include insula models (exclude baseline models)
            return use_insula
                
        except Exception as e:
            print(f"  Warning: Could not read hp.json from {model_path}: {e}")
            return False

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
        """Find checkpoint with best timing error performance for time bisection models"""
        
        best_checkpoint = None
        best_metric = float('inf')  # Lower is better for choice error
        
        print(f"    Searching for best checkpoint based on choice error...")
        
        for checkpoint_num, checkpoint_path in numbered_checkpoints:
            log_path = os.path.join(checkpoint_path, 'log.json')
            
            if not os.path.exists(log_path):
                continue
                
            try:
                with open(log_path, 'r') as f:
                    checkpoint_log = json.load(f)
                
                # Look for appropriate metric for time bisection tasks
                timing_error = None
                if 'mean_choice_error' in checkpoint_log:
                    # For bisection tasks, use choice error (lower is better)
                    timing_error = checkpoint_log['mean_choice_error']
                elif 'cost' in checkpoint_log:
                    # Fallback to cost (lower is better)
                    timing_error = checkpoint_log['cost']
                elif 'success_action_prob' in checkpoint_log:
                    # Convert success probability to error (lower is better)
                    timing_error = 1.0 - checkpoint_log['success_action_prob']
                    
                if timing_error is not None and timing_error < best_metric:
                    best_metric = timing_error
                    best_checkpoint = checkpoint_path
                    print(f"      Checkpoint {checkpoint_num}: choice_error = {timing_error:.6f} (best so far)")
                elif timing_error is not None:
                    print(f"      Checkpoint {checkpoint_num}: choice_error = {timing_error:.6f}")
                    
            except Exception as e:
                print(f"    Warning: Could not read log from {checkpoint_path}: {e}")
                continue
        
        if best_checkpoint:
            print(f"    Selected checkpoint with best choice error: {best_metric:.6f}")
            return best_checkpoint
        else:
            print(f"    Warning: No timing error data found in checkpoint logs")
            return None

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)


def main():
    """Run the cardiac library comparison test for insula models"""

    # Initialize the tester
    tester = TimeBisectionInsulaLibraryTester()

    # Discover time bisection insula models
    model_directories = tester.discover_insula_bisection_models()

    if not model_directories:
        print("No time bisection insula models found!")
        return

    print(f"Found {len(model_directories)} time bisection insula models")

    # Run the comparison across libraries
    results = tester.test_multiple_models_across_libraries(model_directories)

    print(f"\nCardiac library comparison complete!")
    print(f"Results saved to: {tester.base_results_dir}")

    return results


if __name__ == "__main__":
    main()
