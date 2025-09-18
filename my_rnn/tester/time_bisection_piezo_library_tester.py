"""
Time Bisection Cardiac Library Comparison Tester
================================================

Tests time bisection piezo models across different cardiac libraries
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


class TimeBisectionCardiacLibraryTester:
    """Test time bisection models across different cardiac libraries"""

    def __init__(self, base_results_dir="time_bisection_cardiac_library_results"):
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

        print(f"Time Bisection Cardiac Library Tester initialized")
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

    def _calculate_time_bisection_performance_debug(self, trial, outputs, condition, debug_info):
        """Calculate performance with extensive debugging to identify the issue"""

        y = trial.y  # [T, B, O] - targets
        T, B, O = outputs.shape

        # DEBUG: Print shapes and basic info
        debug_info['trial_y_shape'] = y.shape
        debug_info['outputs_shape'] = outputs.shape
        debug_info['T'] = T
        debug_info['B'] = B
        debug_info['O'] = O

        # DEBUG: Check target values
        debug_info['targets_min'] = y.min()
        debug_info['targets_max'] = y.max()
        debug_info['targets_sum_total'] = y.sum()
        debug_info['targets_nonzero_count'] = (y != 0).sum()

        # Find response time from targets
        targets_active = (np.abs(y).sum(axis=2) > 0)  # [T, B]
        has_target = targets_active.any(axis=0)  # [B]

        if not has_target.any():
            debug_info['error'] = 'No active targets found'
            return {'accuracy': 0.0, 'debug_error': 'no_active_targets'}

        response_idx_targets = targets_active.astype(float).argmax(axis=0)  # [B]

        # DEBUG: Response timing
        debug_info['response_idx'] = response_idx_targets.tolist()
        debug_info['has_target'] = has_target.tolist()

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

        # DEBUG: Check what we're comparing
        debug_info['chosen_logits_shape'] = chosen_logits.shape
        debug_info['chosen_targets_shape'] = chosen_targets.shape
        debug_info['chosen_logits_sample'] = chosen_logits[0].tolist() if B > 0 else []
        debug_info['chosen_targets_sample'] = chosen_targets[0].tolist() if B > 0 else []

        # Calculate predictions and accuracy
        pred_classes = np.argmax(chosen_logits, axis=1)  # [B]
        target_classes = np.argmax(chosen_targets, axis=1)  # [B]

        # DEBUG: Check class predictions
        debug_info['pred_classes'] = pred_classes.tolist()
        debug_info['target_classes'] = target_classes.tolist()
        debug_info['pred_class_0_count'] = (pred_classes == 0).sum()
        debug_info['pred_class_1_count'] = (pred_classes == 1).sum()
        debug_info['target_class_0_count'] = (target_classes == 0).sum()
        debug_info['target_class_1_count'] = (target_classes == 1).sum()

        correct = (pred_classes == target_classes).astype(float)
        accuracy = float(np.mean(correct)) if correct.size > 0 else 0.0

        # DEBUG: Final accuracy calculation
        debug_info['correct_predictions'] = correct.tolist()
        debug_info['num_correct'] = correct.sum()
        debug_info['total_predictions'] = len(correct)
        debug_info['calculated_accuracy'] = accuracy

        # CRITICAL DEBUG: Check if the issue is with target generation
        duration = condition.get('duration', 0)
        short_std = condition.get('short_standard', 1000)
        long_std = condition.get('long_standard', 2000)

        # What SHOULD the target be based on dataset logic?
        if 'correct_choice' in condition:
            expected_target_class = condition['correct_choice']
            debug_info['expected_from_dataset'] = expected_target_class
        else:
            # Fallback to closest standard logic
            short_distance = abs(duration - short_std)
            long_distance = abs(duration - long_std)
            expected_target_class = 0 if short_distance < long_distance else 1
            debug_info['expected_from_closest'] = expected_target_class

        # What does the bisection point method give?
        bisection_point = (short_std + long_std) / 2
        bisection_target_class = 0 if duration <= bisection_point else 1
        debug_info['expected_from_bisection'] = bisection_target_class
        debug_info['duration'] = duration
        debug_info['bisection_point'] = bisection_point
        debug_info['short_std'] = short_std
        debug_info['long_std'] = long_std

        # Check if there's a mismatch between expected and actual targets
        if len(target_classes) > 0:
            actual_target_class = target_classes[0]  # First batch element
            debug_info['actual_target_class'] = int(actual_target_class)
            debug_info['target_mismatch'] = actual_target_class != expected_target_class

        return {
            'accuracy': float(accuracy),
            'correct_predictions': int(correct.sum()) if correct.size > 0 else 0,
            'total_predictions': int(correct.size),
            'duration': duration,
            'ground_truth_class': int(target_classes[0]) if len(target_classes) > 0 else -1,
            'predicted_class': int(pred_classes[0]) if len(pred_classes) > 0 else -1
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
                if debug.get('target_mismatch', False):
                    debug_summary['target_mismatches'] += 1

                # Collect prediction distributions
                if 'pred_class_0_count' in debug and 'pred_class_1_count' in debug:
                    debug_summary['pred_class_distribution']['class_0'] += debug['pred_class_0_count']
                    debug_summary['pred_class_distribution']['class_1'] += debug['pred_class_1_count']

                if 'target_class_0_count' in debug and 'target_class_1_count' in debug:
                    debug_summary['target_class_distribution']['class_0'] += debug['target_class_0_count']
                    debug_summary['target_class_distribution']['class_1'] += debug['target_class_1_count']

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
        print(f"      Pred distribution: {debug_summary['pred_class_distribution']}")
        print(f"      Target distribution: {debug_summary['target_class_distribution']}")

        if debug_summary['target_mismatches'] > 0:
            print(f"      WARNING: {debug_summary['target_mismatches']} target mismatches detected!")
            print(f"      This suggests target generation is inconsistent with dataset expectations")

        return summary

    def _summarize_performance_fixed(self, test_results):
        """FIXED: Summarize performance using the same logic as the working tester"""

        if not test_results:
            return {}

        # FIXED: Use the same filtering logic as interval_production_tester.py
        # Don't filter by has_nan - only filter by actual NaN values in the metrics
        accuracies = [r['performance']['accuracy'] for r in test_results
                      if 'accuracy' in r['performance'] and not np.isnan(r['performance']['accuracy'])]

        costs = [r['cost'] for r in test_results if not np.isnan(r['cost']) and np.isfinite(r['cost'])]

        # Calculate summary with better error handling
        summary = {
            'num_conditions': len(test_results),
            'num_valid': len(accuracies),
            'nan_rate': (len(test_results) - len(accuracies)) / len(test_results) if test_results else 0,
            'accuracy': np.mean(accuracies) if accuracies else 0,  # FIXED: Don't use nanmean on empty list
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'mean_cost': np.mean(costs) if costs else np.inf
        }

        # Debug output to see what's happening
        print(f"    Summary: {len(accuracies)} valid accuracies out of {len(test_results)} conditions")
        if accuracies:
            print(f"    Mean accuracy: {summary['accuracy']:.4f}")
        else:
            print(f"    No valid accuracies found!")

        return summary

    def _calculate_time_bisection_performance(self, trial, outputs, condition):
        """Calculate time bisection performance with target mismatch debugging"""

        y = trial.y  # [T, B, O] - targets from task generation
        T, B, O = outputs.shape

        # Find response timing (existing logic)
        targets_active = (np.abs(y).sum(axis=2) > 0)
        has_target = targets_active.any(axis=0)
        response_idx = targets_active.astype(float).argmax(axis=0)

        if not has_target.any():
            return {'accuracy': 0.0, 'debug_error': 'no_active_targets'}

        # Get outputs and targets at response time
        logits_b = np.transpose(outputs, (1, 0, 2))
        targets_b = np.transpose(y, (1, 0, 2))
        rows = np.arange(B)
        chosen_logits = logits_b[rows, response_idx, :]
        chosen_targets = targets_b[rows, response_idx, :]

        # Calculate predictions
        pred_classes = np.argmax(chosen_logits, axis=1)
        target_classes = np.argmax(chosen_targets, axis=1)

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
            f.write("TIME BISECTION CARDIAC LIBRARY COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Models Tested: {len(all_results)}\n")
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
            library_stats = df.groupby('library').agg({
                'accuracy': ['mean', 'std', 'count'],
                'mean_cost': 'mean',
                'nan_rate': 'mean'
            }).round(4)

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
            f.write("\n")

            # Model Consistency Analysis
            f.write("MODEL CONSISTENCY ANALYSIS\n")
            f.write("-" * 35 + "\n")
            model_stats = df.groupby('model').agg({
                'accuracy': ['mean', 'std'],
                'library': 'count'
            }).round(4)

            f.write("Models ranked by consistency (Lower Std = More Consistent across libraries):\n\n")
            model_consistency = df.groupby('model')['accuracy'].std().sort_values()

            for rank, (model_name, std_accuracy) in enumerate(model_consistency.items(), 1):
                model_data = df[df['model'] == model_name]
                mean_accuracy = model_data['accuracy'].mean()
                count = len(model_data)
                min_perf = model_data['accuracy'].min()
                max_perf = model_data['accuracy'].max()

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
                f.write(f"  Models Tested: {len(lib_data)}\n")
                f.write(
                    f"  Mean Accuracy: {lib_data['accuracy'].mean():.4f} ± {lib_data['accuracy'].std():.4f}\n")
                f.write(
                    f"  Accuracy Range: [{lib_data['accuracy'].min():.4f}, {lib_data['accuracy'].max():.4f}]\n")
                f.write(f"  Mean Cost: {lib_data['mean_cost'].mean():.4f}\n")
                f.write(f"  Average NaN Rate: {lib_data['nan_rate'].mean():.1%}\n")
                # Add ID/OOD accuracy reporting
                if 'mean_id_accuracy' in lib_data:
                    f.write(f"  Mean In-Distribution Accuracy: {lib_data['mean_id_accuracy'].mean():.4f}\n")
                if 'mean_ood_accuracy' in lib_data:
                    f.write(f"  Mean Out-of-Distribution Accuracy: {lib_data['mean_ood_accuracy'].mean():.4f}\n")

                # Best and worst models for this library
                best_model = lib_data.loc[lib_data['accuracy'].idxmax()]
                worst_model = lib_data.loc[lib_data['accuracy'].idxmin()]
                f.write(f"  Best Model: {best_model['model']} ({best_model['accuracy']:.4f})\n")
                f.write(f"  Worst Model: {worst_model['model']} ({worst_model['accuracy']:.4f})\n")

            # Detailed Model Analysis
            f.write("\n\nDETAILED MODEL ANALYSIS\n")
            f.write("-" * 28 + "\n")

            for model_name in sorted(df['model'].unique()):
                model_data = df[df['model'] == model_name]

                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Libraries Tested: {len(model_data)}\n")
                f.write(
                    f"  Mean Accuracy: {model_data['accuracy'].mean():.4f} ± {model_data['accuracy'].std():.4f}\n")
                f.write(
                    f"  Accuracy Range: [{model_data['accuracy'].min():.4f}, {model_data['accuracy'].max():.4f}]\n")
                f.write(f"  Average NaN Rate: {model_data['nan_rate'].mean():.1%}\n")

                # Best and worst libraries for this model
                best_library = model_data.loc[model_data['accuracy'].idxmax()]
                worst_library = model_data.loc[model_data['accuracy'].idxmin()]
                f.write(f"  Best Library: {best_library['library']} ({best_library['accuracy']:.4f})\n")
                f.write(f"  Worst Library: {worst_library['library']} ({worst_library['accuracy']:.4f})\n")
                f.write(f"  Library Sensitivity: {model_data['accuracy'].std():.4f} (lower = more robust)\n")

            # Statistical Analysis
            f.write("\n\nSTATISTICAL ANALYSIS\n")
            f.write("-" * 25 + "\n")

            # ANOVA-like analysis
            total_variance = df['accuracy'].var()
            between_library_variance = df.groupby('library')['accuracy'].mean().var()
            between_model_variance = df.groupby('model')['accuracy'].mean().var()

            f.write(f"Total Performance Variance: {total_variance:.6f}\n")
            f.write(
                f"Between-Library Variance: {between_library_variance:.6f} ({between_library_variance / total_variance:.1%} of total)\n")
            f.write(
                f"Between-Model Variance: {between_model_variance:.6f} ({between_model_variance / total_variance:.1%} of total)\n")

            # Correlation analysis
            f.write(f"\nPerformance Correlations:\n")
            f.write(f"  Accuracy vs Cost: {df['accuracy'].corr(df['mean_cost']):.3f}\n")
            f.write(f"  Accuracy vs NaN Rate: {df['accuracy'].corr(df['nan_rate']):.3f}\n")

            # Top and bottom performing combinations
            f.write(f"\n\nTOP 10 MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 40 + "\n")
            top_combinations = df.nlargest(10, 'accuracy')
            for i, row in top_combinations.iterrows():
                f.write(
                    f"{row.name + 1:2d}. {row['model']:20s} + {row['library']:15s} | {row['accuracy']:.4f}\n")

            f.write(f"\n\nBOTTOM 10 MODEL-LIBRARY COMBINATIONS\n")
            f.write("-" * 42 + "\n")
            bottom_combinations = df.nsmallest(10, 'accuracy')
            for i, row in bottom_combinations.iterrows():
                f.write(
                    f"{len(df) - row.name:2d}. {row['model']:20s} + {row['library']:15s} | {row['accuracy']:.4f}\n")

            # Recommendations
            f.write(f"\n\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")

            best_overall_library = library_means.index[0]
            most_consistent_library = df.groupby('library')['accuracy'].std().idxmin()
            most_robust_model = model_consistency.index[0]
            best_overall_model = df.groupby('model')['accuracy'].mean().idxmax()

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
            f.write(f"- Accuracy: 1.0 = perfect classification, 0.5 = chance level\n")
            f.write(f"- Training range: 1000-2000ms durations\n")
            f.write(f"- Bisection point: 600ms (short < 600ms < long)\n")
            f.write(f"- NaN rate: Proportion of trials producing invalid outputs\n")
            f.write(f"- Library CV: Coefficient of variation of library means\n")
            f.write(f"- Model sensitivity: Standard deviation across libraries\n")
            f.write(
                f"- Total conditions tested per model-library pair: {df['num_conditions'].iloc[0] if len(df) > 0 else 'N/A'}\n")

            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")

        print(f"Comprehensive written report saved to: {report_path}")
        return report_path

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
        fig.suptitle('Time Bisection: Cardiac Library Comparison', fontsize=16)

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

    def discover_bisection_models(self):
        """Discover trained time bisection models"""
        model_dirs = []

        possible_base_dirs = [
            'enhanced_piezo_time_bisection_results',  # FIXED: Added the correct directory name
            'enhanced_piezo_bisection_results',
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

    def _find_best_model_checkpoint(self, base_model_dir):
        """Find the best model checkpoint in a run directory"""
        final_path = os.path.join(base_model_dir, 'finalResult')
        if os.path.exists(final_path) and self._has_valid_model(final_path):
            return final_path

        numbered_checkpoints = []
        if os.path.exists(base_model_dir):
            for item in os.listdir(base_model_dir):
                if item.isdigit():
                    checkpoint_path = os.path.join(base_model_dir, item)
                    if os.path.isdir(checkpoint_path) and self._has_valid_model(checkpoint_path):
                        numbered_checkpoints.append((int(item), checkpoint_path))

        if numbered_checkpoints:
            numbered_checkpoints.sort(reverse=True)
            return numbered_checkpoints[0][1]

        if self._has_valid_model(base_model_dir):
            return base_model_dir

        return None

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)


def main():
    """Run the cardiac library comparison test"""

    # Initialize the tester
    tester = TimeBisectionCardiacLibraryTester()

    # Discover time bisection models
    model_directories = tester.discover_bisection_models()

    if not model_directories:
        print("No time bisection models found!")
        return

    print(f"Found {len(model_directories)} time bisection models")

    # Run the comparison across libraries
    results = tester.test_multiple_models_across_libraries(model_directories)

    print(f"\nCardiac library comparison complete!")
    print(f"Results saved to: {tester.base_results_dir}")

    return results


if __name__ == "__main__":
    main()