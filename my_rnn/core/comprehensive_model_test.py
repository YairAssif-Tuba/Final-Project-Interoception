"""
Comprehensive Model Testing with Task Visualization and Performance Comparison - ENHANCED VERSION
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from collections import defaultdict
import pandas as pd

# Add project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import Runner
import tools
import task
import default
from real_cardiac_data import create_real_cardiac_data_for_task


class ComprehensiveModelTester:
    """Comprehensive tester for trained RNN models with task visualization"""

    def __init__(self, base_results_dir="test_results_comprehensive"):
        self.base_results_dir = base_results_dir
        self.cardiac_data_path = "/Users/itanarbuldaliot/Documents/prj_interoception_modeling/10shrv0"
        tools.mkdir_p(base_results_dir)

        # Enhanced intervals from your training
        self.enhanced_intervals = {
            'min_interval': 1200,  # 1.2 seconds minimum
            'max_interval': 2400,  # 2.4 seconds maximum
            'delay_interval': 1000  # 1 second delay
        }

        print(f"Comprehensive Model Tester initialized")
        print(f"Results directory: {self.base_results_dir}")
        print(f"Cardiac data path: {self.cardiac_data_path}")
        print(
            f"Enhanced intervals: {self.enhanced_intervals['min_interval']}-{self.enhanced_intervals['max_interval']}ms")

    def test_single_model(self, model_dir, rule_name, test_conditions):
        """Test a single model comprehensively"""
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

        # Get model info (fixed method call)
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
            'nan_counts_per_condition': []  # NEW: Track NaN counts per condition
        }

        # Test each condition
        for i, condition in enumerate(test_conditions):
            print(f"  Testing condition {i + 1}/{len(test_conditions)}: {condition}")

            try:
                # Generate cardiac data if piezo model
                cardiac_data = None
                if use_piezo:
                    # Calculate task duration for this condition
                    if rule_name == 'interval_comparison':
                        task_duration_ms = max(condition['prod_interval1'], condition['prod_interval2']) + condition[
                            'dly_interval'] + 2000
                    elif rule_name == 'interval_production':
                        task_duration_ms = condition['prod_interval'] * 2 + condition['dly_interval'] + 2000
                    elif rule_name == 'time_bisection':
                        # Use the longest standard + response window + small buffer
                        ls = int(condition.get('long_standard', 900))
                        resp = int(condition.get('response_duration', 300))
                        task_duration_ms = ls + resp + 1000  # 1s buffer

                    from dataset import TaskDataset

                    try:
                        import default
                        # Start with a full set of defaults (includes 'dt', etc.)
                        hp = default.get_default_hp(rule_name=rule_name, use_piezo=model_info.get("use_piezo", False))
                        # Update with model-specific info from saved model
                        hp.update(model_info)

                        # Build dataset sample with real cardiac data
                        ds = TaskDataset(rule_name, hp, mode="test")
                        sample = ds[0]
                        condition['hb_sequence'] = sample['hb_sequence']
                        print("    Loaded real cardiac data from dataset")
                    except Exception as e:
                        print(f"    Could not load cardiac data from dataset: {e}")
                    except Exception as e:
                        print(f"    Warning: Could not generate cardiac data: {e}")
                        print(f"    Testing without cardiac data")

                trial, train_stepper = runner.run(**condition)

                # Extract comprehensive results and check for NaN
                outputs = train_stepper.outputs.detach().cpu().numpy()
                states = [state.detach().cpu().numpy() for state in train_stepper.state_collector]
                cost = train_stepper.cost.item()

                # Check for NaN values in outputs, states, cost
                has_nan_outputs = np.isnan(outputs).any()
                has_nan_states = any(np.isnan(state).any() for state in states)
                has_nan_cost = np.isnan(cost)

                # Calculate performance first
                performance = self._calculate_performance(rule_name, trial, outputs)

                # ADDED: Check performance metrics for NaN
                performance_has_nan = False
                if 'mean_completion' in performance:
                    if np.isnan(performance['mean_completion']):
                        performance_has_nan = True
                if 'mean_timing_error' in performance:
                    if not np.isfinite(performance['mean_timing_error']):
                        performance_has_nan = True
                if 'accuracy' in performance:
                    if np.isnan(performance['accuracy']):
                        performance_has_nan = True

                # Update the overall NaN detection
                has_nan = has_nan_outputs or has_nan_states or has_nan_cost or performance_has_nan

                if has_nan:
                    print(f"    WARNING: NaN detected in condition {i + 1}")
                    print(f"      NaN in outputs: {has_nan_outputs}")
                    print(f"      NaN in states: {has_nan_states}")
                    print(f"      NaN in cost: {has_nan_cost}")
                    print(f"      NaN in performance: {performance_has_nan}")

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
                    'performance': performance,  # Use the pre-calculated performance
                    'has_nan': has_nan,  # Updated NaN detection
                    'nan_details': {  # Updated NaN details
                        'outputs': has_nan_outputs,
                        'states': has_nan_states,
                        'cost': has_nan_cost,
                        'performance': performance_has_nan  # Added performance NaN tracking
                    }
                }

                results['test_results'].append(test_result)
                results['nan_counts_per_condition'].append(1 if has_nan else 0)
            except Exception as e:
                print(f"    Error testing condition {condition}: {e}")
                # Track as NaN condition due to failure
                results['nan_counts_per_condition'].append(1)
                import traceback
                traceback.print_exc()
                continue

        # Calculate performance summary
        if results['test_results']:
            results['performance_summary'] = self._summarize_performance(results['test_results'])
            # NEW: Add NaN summary
            results['performance_summary']['total_nan_conditions'] = sum(results['nan_counts_per_condition'])
            results['performance_summary']['total_conditions'] = len(test_conditions)
            results['performance_summary']['nan_rate'] = sum(results['nan_counts_per_condition']) / len(test_conditions)

        return results

    def _calculate_performance(self, rule_name, trial, outputs):
        """Calculate task-specific performance metrics"""
        performance = {}

        if rule_name == 'interval_comparison':
            # Extract choice from outputs
            batch_size = outputs.shape[1]

            # Find response epoch - use 'go' epoch which should be the response period
            if 'go' in trial.epochs:
                response_start = trial.epochs['go'][0][0] if hasattr(trial.epochs['go'][0], '__iter__') else \
                    trial.epochs['go'][0]
            else:
                # Fallback: use last 20% of sequence for response
                response_start = int(0.8 * outputs.shape[0])

            # Get final outputs (average over response period for stability)
            response_outputs = outputs[response_start:, :, :]
            final_outputs = np.mean(response_outputs, axis=0)  # Average over time steps

            # Choice is which output unit has higher activation
            choices = np.argmax(final_outputs, axis=1)  # 0 or 1

            # FIXED: Ground truth logic
            # If interval 1 > interval 2, correct choice is 0 (first output)
            # If interval 1 <= interval 2, correct choice is 1 (second output)
            correct_choices = (trial.prod_interval1 <= trial.prod_interval2).astype(int)

            # Calculate accuracy
            accuracy = np.mean(choices == correct_choices)

            performance = {
                'accuracy': accuracy,
                'choices': choices.tolist(),
                'correct_choices': correct_choices.tolist(),
                'final_outputs': final_outputs.tolist(),
                'interval1': trial.prod_interval1.tolist() if hasattr(trial.prod_interval1, 'tolist') else [
                    trial.prod_interval1],
                'interval2': trial.prod_interval2.tolist() if hasattr(trial.prod_interval2, 'tolist') else [
                    trial.prod_interval2],
                'response_start_time': response_start
            }

        elif rule_name == 'time_bisection':
            # Classification at the response moment (two classes)
            # We have access to trial.y (targets) and trial.cost_mask in the saved trial dict
            # outputs: [T, B, O], trial.y: [T, B, O], cost_mask: [T, B] or [T, B, O]
            y = trial.y
            T, B, O = outputs.shape

            # 1) Find response index per trial: first timestep where targets become non-zero
            #    (robust to all-zero targets by falling back to first masked step)
            targets_active = (np.abs(y).sum(axis=2) > 0)  # [T, B]
            has_target = targets_active.any(axis=0)  # [B]
            response_idx_targets = targets_active.astype(float).argmax(axis=0)  # [B]

            # Mask fallback (works if mask is [T,B] or [T,B,O])
            cm = trial.cost_mask
            if cm.ndim == 3:
                mask_tb = (cm != 0).any(axis=2)  # [T, B]
            else:
                mask_tb = cm  # [T, B]
            response_idx_mask = mask_tb.astype(float).argmax(axis=0)  # [B]

            response_idx = np.where(has_target, response_idx_targets, response_idx_mask)  # [B]

            # 2) Gather logits and targets at response step
            #    Convert to batch-first arrays for easy indexing
            logits_b = np.transpose(outputs, (1, 0, 2))  # [B, T, O]
            targets_b = np.transpose(y, (1, 0, 2))  # [B, T, O]

            rows = np.arange(B)
            chosen_logits = logits_b[rows, response_idx, :]  # [B, O]
            chosen_targets = targets_b[rows, response_idx, :]  # [B, O]

            # 3) Pred / target classes
            pred_classes = np.argmax(chosen_logits, axis=1)  # [B]
            target_classes = np.argmax(chosen_targets, axis=1)  # [B]

            # 4) Metrics
            correct = (pred_classes == target_classes).astype(float)
            accuracy = float(np.mean(correct)) if correct.size else float('nan')

            performance = {
                'accuracy': accuracy,
                'pred_classes': pred_classes,
                'target_classes': target_classes,
                'response_indices': response_idx,
                'response_start_time': int(response_idx.min()) if response_idx.size else None
            }

        elif rule_name == 'interval_production':

            # For production task, check timing accuracy

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
                    error_steps = abs(produced_steps - target)  # absolute error in steps
                    ratio = produced_steps / target
                    symmetric_completion = 1 - abs(ratio - 1)  # % of completion (can be <1, =1, or >1)

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
                # New metric: completion ratios
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

        # Determine rule type by checking what performance metrics exist
        first_perf = test_results[0]['performance']

        if 'accuracy' in first_perf:
            # Interval comparison task
            accuracies = [r['performance']['accuracy'] for r in test_results if 'accuracy' in r['performance']]
            if accuracies:
                summary.update({
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'task_type': test_results[0].get('rule_name', 'interval_comparison')
                })
        elif 'mean_timing_error' in first_perf:
            # Interval production task
            timing_errors = [r['performance']['mean_timing_error'] for r in test_results if
                             'mean_timing_error' in r['performance']]
            completions = [r['performance']['mean_completion'] for r in test_results if
                           'mean_completion' in r['performance']]
            if timing_errors and completions:
                summary.update({
                    'mean_timing_error': np.nanmean(timing_errors),
                    'std_timing_error': np.nanstd(timing_errors),
                    'mean_completion': np.nanmean(completions),
                    'std_completion': np.nanstd(completions),
                    'task_type': 'interval_production'
                })

        return summary

    def plot_condition_consolidated_graphs(self, all_results):
        """NEW: Create consolidated graphs for each condition across models"""
        print("Creating per-condition consolidated graphs...")

        # Group results by rule and condition
        condition_data = defaultdict(lambda: defaultdict(list))

        for result in all_results:
            rule_name = result['rule_name']
            for test_result in result['test_results']:
                condition_idx = test_result['condition_idx']
                condition_key = f"{rule_name}_condition_{condition_idx + 1}"

                condition_data[condition_key]['results'].append({
                    'model_name': result['model_name'],
                    'use_piezo': result['use_piezo'],
                    'performance': test_result['performance'],
                    'cost': test_result['cost'],
                    'has_nan': test_result['has_nan'],
                    'condition': test_result['condition']
                })

        # Create plots for each condition
        for condition_key, data in condition_data.items():
            self._plot_single_condition_comparison(condition_key, data['results'])

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

        # Plot 1: Performance comparison
        ax1 = axes[0, 0]
        if condition_results and 'accuracy' in condition_results[0]['performance']:
            # Accuracy comparison for interval comparison tasks
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

        elif condition_results and 'mean_timing_error' in condition_results[0]['performance']:
            # FIXED: Use completion ratio for production tasks
            if piezo_models:
                # FIXED: Better NaN handling and use completion ratios
                piezo_completions = [r['performance']['mean_completion'] for r in piezo_models
                                     if not r['has_nan'] and 'mean_completion' in r['performance']
                                     and not np.isnan(r['performance']['mean_completion'])]
                if piezo_completions:
                    ax1.bar(['Piezo'], [np.nanmean(piezo_completions)], yerr=[np.nanstd(piezo_completions)],
                            alpha=0.7, color='red', capsize=5)
                    ax1.scatter(['Piezo'] * len(piezo_completions), piezo_completions, color='darkred', s=50, zorder=10)

            if no_piezo_models:
                no_piezo_completions = [r['performance']['mean_completion'] for r in no_piezo_models
                                        if not r['has_nan'] and 'mean_completion' in r['performance']
                                        and not np.isnan(r['performance']['mean_completion'])]
                if no_piezo_completions:
                    ax1.bar(['No Piezo'], [np.nanmean(no_piezo_completions)], yerr=[np.nanstd(no_piezo_completions)],
                            alpha=0.7, color='blue', capsize=5)
                    ax1.scatter(['No Piezo'] * len(no_piezo_completions), no_piezo_completions, color='darkblue', s=50,
                                zorder=10)

            ax1.set_ylabel('Completion Ratio')
            ax1.set_title('Performance Comparison (Completion Ratio)')
            ax1.set_ylim(0, 2.0)  # Allow for completion ratios > 1.0

        ax1.grid(True, alpha=0.3)

        # Plot 2: FIXED - Show timing error as secondary metric for production tasks
        ax2 = axes[0, 1]
        if condition_results and 'mean_timing_error' in condition_results[0]['performance']:
            if piezo_models:
                # FIXED: Proper NaN filtering and use nanmean
                piezo_errors = [r['performance']['mean_timing_error'] for r in piezo_models
                                if not r['has_nan'] and 'mean_timing_error' in r['performance']
                                and np.isfinite(r['performance']['mean_timing_error'])]
                if piezo_errors:
                    ax2.bar(['Piezo'], [np.nanmean(piezo_errors)], yerr=[np.nanstd(piezo_errors)],
                            alpha=0.7, color='red', capsize=5)
                    ax2.scatter(['Piezo'] * len(piezo_errors), piezo_errors, color='darkred', s=50, zorder=10)

            if no_piezo_models:
                no_piezo_errors = [r['performance']['mean_timing_error'] for r in no_piezo_models
                                   if not r['has_nan'] and 'mean_timing_error' in r['performance']
                                   and np.isfinite(r['performance']['mean_timing_error'])]
                if no_piezo_errors:
                    ax2.bar(['No Piezo'], [np.nanmean(no_piezo_errors)], yerr=[np.nanstd(no_piezo_errors)],
                            alpha=0.7, color='blue', capsize=5)
                    ax2.scatter(['No Piezo'] * len(no_piezo_errors), no_piezo_errors, color='darkblue', s=50, zorder=10)

            ax2.set_ylabel('Timing Error (steps)')
            ax2.set_title('Timing Error Comparison')

        else:
            # FIXED: For comparison tasks, show cost comparison
            if piezo_models:
                piezo_costs = [r['cost'] for r in piezo_models if not r['has_nan'] and np.isfinite(r['cost'])]
                if piezo_costs:
                    ax2.bar(['Piezo'], [np.nanmean(piezo_costs)], yerr=[np.nanstd(piezo_costs)],
                            alpha=0.7, color='red', capsize=5)
                    ax2.scatter(['Piezo'] * len(piezo_costs), piezo_costs, color='darkred', s=50, zorder=10)

            if no_piezo_models:
                no_piezo_costs = [r['cost'] for r in no_piezo_models if not r['has_nan'] and np.isfinite(r['cost'])]
                if no_piezo_costs:
                    ax2.bar(['No Piezo'], [np.nanmean(no_piezo_costs)], yerr=[np.nanstd(no_piezo_costs)],
                            alpha=0.7, color='blue', capsize=5)
                    ax2.scatter(['No Piezo'] * len(no_piezo_costs), no_piezo_costs, color='darkblue', s=50, zorder=10)

            ax2.set_ylabel('Cost')
            ax2.set_title('Training Cost Comparison')

        ax2.grid(True, alpha=0.3)

        # Plot 3: Model count and NaN summary (unchanged)
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

        # Plot 4: Condition details (unchanged)
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
    def compare_model_performance(self, all_results, save_path=None):
        """Compare performance across all tested models - ENHANCED with NaN tracking"""
        if not all_results:
            print("No results to compare")
            return

        # Separate by task type
        comparison_results = [r for r in all_results if r['rule_name'] == 'interval_comparison']
        production_results = [r for r in all_results if r['rule_name'] == 'interval_production']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Added extra column for NaN tracking

        # Comparison task analysis
        if comparison_results:
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            ax3 = axes[0, 2]  # NEW: NaN tracking

            piezo_models = [r for r in comparison_results if r['use_piezo']]
            no_piezo_models = [r for r in comparison_results if not r['use_piezo']]

            # Accuracy comparison
            if piezo_models:
                piezo_accs = [r['performance_summary']['mean_accuracy'] for r in piezo_models
                              if 'mean_accuracy' in r['performance_summary']]
                if piezo_accs:
                    ax1.bar(['Piezo'], [np.mean(piezo_accs)],
                            yerr=[np.std(piezo_accs)], alpha=0.7, color='red',
                            capsize=5, label='Piezo Models')
                    ax1.scatter(['Piezo'] * len(piezo_accs), piezo_accs,
                                color='darkred', s=50, alpha=0.8, zorder=10)

            if no_piezo_models:
                no_piezo_accs = [r['performance_summary']['mean_accuracy'] for r in no_piezo_models
                                 if 'mean_accuracy' in r['performance_summary']]
                if no_piezo_accs:
                    ax1.bar(['No Piezo'], [np.mean(no_piezo_accs)],
                            yerr=[np.std(no_piezo_accs)], alpha=0.7, color='blue',
                            capsize=5, label='Standard Models')
                    ax1.scatter(['No Piezo'] * len(no_piezo_accs), no_piezo_accs,
                                color='darkblue', s=50, alpha=0.8, zorder=10)

            ax1.set_title('Interval Comparison - Accuracy')
            ax1.set_ylabel('Mean Accuracy')
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3)

            # Cost comparison
            if piezo_models and no_piezo_models:
                piezo_costs = [r['performance_summary']['mean_cost'] for r in piezo_models]
                no_piezo_costs = [r['performance_summary']['mean_cost'] for r in no_piezo_models]

                ax2.bar(['Piezo', 'No Piezo'],
                        [np.mean(piezo_costs), np.mean(no_piezo_costs)],
                        yerr=[np.std(piezo_costs), np.std(no_piezo_costs)],
                        color=['red', 'blue'], alpha=0.7, capsize=5)

                ax2.scatter(['Piezo'] * len(piezo_costs), piezo_costs,
                            color='darkred', s=50, alpha=0.8, zorder=10)
                ax2.scatter(['No Piezo'] * len(no_piezo_costs), no_piezo_costs,
                            color='darkblue', s=50, alpha=0.8, zorder=10)

            ax2.set_title('Interval Comparison - Training Cost')
            ax2.set_ylabel('Mean Cost')
            ax2.grid(True, alpha=0.3)

            # NEW: NaN rate comparison
            if piezo_models:
                piezo_nan_rates = [r['performance_summary'].get('nan_rate', 0) for r in piezo_models]
                ax3.bar(['Piezo'], [np.mean(piezo_nan_rates)],
                        yerr=[np.std(piezo_nan_rates)], alpha=0.7, color='red', capsize=5)
                ax3.scatter(['Piezo'] * len(piezo_nan_rates), piezo_nan_rates,
                            color='darkred', s=50, alpha=0.8, zorder=10)

            if no_piezo_models:
                no_piezo_nan_rates = [r['performance_summary'].get('nan_rate', 0) for r in no_piezo_models]
                ax3.bar(['No Piezo'], [np.mean(no_piezo_nan_rates)],
                        yerr=[np.std(no_piezo_nan_rates)], alpha=0.7, color='blue', capsize=5)
                ax3.scatter(['No Piezo'] * len(no_piezo_nan_rates), no_piezo_nan_rates,
                            color='darkblue', s=50, alpha=0.8, zorder=10)

            ax3.set_title('Interval Comparison - NaN Rate')
            ax3.set_ylabel('NaN Rate')
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)

        # Production task analysis
        if production_results:
            ax4 = axes[1, 0]
            ax5 = axes[1, 1]
            ax6 = axes[1, 2]  # NEW: NaN tracking

            piezo_models = [r for r in production_results if r['use_piezo']]
            no_piezo_models = [r for r in production_results if not r['use_piezo']]

            # Timing error comparison
            if piezo_models:
                piezo_errors = [r['performance_summary']['mean_timing_error'] for r in piezo_models
                                if 'mean_timing_error' in r['performance_summary']]
                if piezo_errors:
                    ax4.bar(['Piezo'], [np.mean(piezo_errors)],
                            yerr=[np.std(piezo_errors)], alpha=0.7, color='red',
                            capsize=5, label='Piezo Models')
                    ax4.scatter(['Piezo'] * len(piezo_errors), piezo_errors,
                                color='darkred', s=50, alpha=0.8, zorder=10)

            if no_piezo_models:
                no_piezo_errors = [r['performance_summary']['mean_timing_error'] for r in no_piezo_models
                                   if 'mean_timing_error' in r['performance_summary']]
                if no_piezo_errors:
                    ax4.bar(['No Piezo'], [np.mean(no_piezo_errors)],
                            yerr=[np.std(no_piezo_errors)], alpha=0.7, color='blue',
                            capsize=5, label='Standard Models')
                    ax4.scatter(['No Piezo'] * len(no_piezo_errors), no_piezo_errors,
                                color='darkblue', s=50, alpha=0.8, zorder=10)

            ax4.set_title('Interval Production - Timing Error')
            ax4.set_ylabel('Mean Relative Timing Error')
            ax4.grid(True, alpha=0.3)

            # Success rate comparison
            if piezo_models and no_piezo_models:
                piezo_completion = [r['performance_summary']['mean_completion'] for r in piezo_models
                                    if 'mean_completion' in r['performance_summary']]
                no_piezo_completion = [r['performance_summary']['mean_completion'] for r in no_piezo_models
                                       if 'mean_completion' in r['performance_summary']]

                if piezo_completion and no_piezo_completion:
                    ax5.bar(['Piezo', 'No Piezo'],
                            [np.nanmean(piezo_completion), np.nanmean(no_piezo_completion)],
                            yerr=[np.nanstd(piezo_completion), np.nanstd(no_piezo_completion)],
                            color=['red', 'blue'], alpha=0.7, capsize=5)

                    ax5.scatter(['Piezo'] * len(piezo_completion), piezo_completion,
                                color='darkred', s=50, alpha=0.8, zorder=10)
                    ax5.scatter(['No Piezo'] * len(no_piezo_completion), no_piezo_completion,
                                color='darkblue', s=50, alpha=0.8, zorder=10)

            ax5.set_title('Interval Production - Mean Completion')
            ax5.set_ylabel('Mean Completion Ratio')
            ax5.set_ylim(0, 1.5)  # allow >1.0 if overshoot responses
            ax5.grid(True, alpha=0.3)

            # NEW: NaN rate comparison for production task
            if piezo_models:
                piezo_nan_rates = [r['performance_summary'].get('nan_rate', 0) for r in piezo_models]
                ax6.bar(['Piezo'], [np.mean(piezo_nan_rates)],
                        yerr=[np.std(piezo_nan_rates)], alpha=0.7, color='red', capsize=5)
                ax6.scatter(['Piezo'] * len(piezo_nan_rates), piezo_nan_rates,
                            color='darkred', s=50, alpha=0.8, zorder=10)

            if no_piezo_models:
                no_piezo_nan_rates = [r['performance_summary'].get('nan_rate', 0) for r in no_piezo_models]
                ax6.bar(['No Piezo'], [np.mean(no_piezo_nan_rates)],
                        yerr=[np.std(no_piezo_nan_rates)], alpha=0.7, color='blue', capsize=5)
                ax6.scatter(['No Piezo'] * len(no_piezo_nan_rates), no_piezo_nan_rates,
                            color='darkblue', s=50, alpha=0.8, zorder=10)

            ax6.set_title('Interval Production - NaN Rate')
            ax6.set_ylabel('NaN Rate')
            ax6.set_ylim(0, 1.1)
            ax6.grid(True, alpha=0.3)

        plt.suptitle('Model Performance Comparison (Enhanced with NaN Tracking)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot: {save_path}")

        plt.show()

    def run_comprehensive_test(self, model_directories=None):
        """Run comprehensive test on specified models - ENHANCED with per-condition graphs and NaN tracking"""
        if model_directories is None:
            # Auto-discover model directories
            model_directories = self._discover_model_directories()

        print(f"Running comprehensive test on {len(model_directories)} models")

        # Define test conditions
        comparison_conditions = [
            {'prod_interval1': 1200, 'prod_interval2': 1400, 'dly_interval': 1000},
            {'prod_interval1': 1000, 'prod_interval2': 1800, 'dly_interval': 1000},
            {'prod_interval1': 1600, 'prod_interval2': 2000, 'dly_interval': 1000},
            {'prod_interval1': 2000, 'prod_interval2': 2400, 'dly_interval': 1000},
        ]

        production_conditions = [
            {'prod_interval': 1200, 'dly_interval': 1000},
            {'prod_interval': 1600, 'dly_interval': 1000},
            {'prod_interval': 2000, 'dly_interval': 1000},
            {'prod_interval': 2400, 'dly_interval': 1000},
        ]
        bisection_conditions = [
            {"short_standard": 300, "long_standard": 900, "std": 40, "response_duration": 300},
            {"short_standard": 400, "long_standard": 1000, "std": 50, "response_duration": 300},
            {"short_standard": 500, "long_standard": 1100, "std": 60, "response_duration": 300},
            {"short_standard": 600, "long_standard": 1200, "std": 70, "response_duration": 300},
        ]

        all_results = []

        for model_info in model_directories:
            if len(model_info) == 3:
                model_dir, rule_name, run_name = model_info
            else:
                model_dir, rule_name = model_info
                run_name = os.path.basename(model_dir)

            if rule_name == 'interval_comparison':
                conditions = comparison_conditions
            elif rule_name == 'interval_production':
                conditions = production_conditions
            elif rule_name == 'time_bisection':
                conditions = bisection_conditions
            else:
                raise ValueError(f"Unknown rule_name: {rule_name}")
            print(f"\nTesting {run_name} ({rule_name})")
            results = self.test_single_model(model_dir, rule_name, conditions)

            if results:
                # Override model_name to use run_name instead of checkpoint directory
                results['model_name'] = run_name
                all_results.append(results)

                # Plot individual model results (separate files)
                self.plot_model_results_separate(results)

                # Save detailed results
                results_path = os.path.join(self.base_results_dir, f"{run_name}_{rule_name}_data.pkl")
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Saved detailed results: {results_path}")

        # Generate comparison plots
        if all_results:
            comparison_path = os.path.join(self.base_results_dir, "model_comparison.png")
            self.compare_model_performance(all_results, comparison_path)

            # NEW: Generate per-condition consolidated graphs
            self.plot_condition_consolidated_graphs(all_results)
            self.plot_interval_length_vs_completion(all_results)

            # Save summary
            summary_path = os.path.join(self.base_results_dir, "test_summary.pkl")
            with open(summary_path, 'wb') as f:
                pickle.dump(all_results, f)

            self._generate_summary_report(all_results)

        return all_results

    def plot_task_structure(self, test_result, model_name, save_dir):
        """Plot task structure separately"""
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

    def plot_model_outputs(self, test_result, model_name, save_dir):
        """Plot model outputs vs targets separately"""
        condition = test_result['condition']
        trial = test_result['trial']
        outputs = test_result['outputs']
        condition_idx = test_result['condition_idx']
        performance = test_result['performance']

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot model outputs
        for ch in range(outputs.shape[2]):
            ax.plot(outputs[:, 0, ch], label=f'Model Output {ch}', linewidth=2)
            ax.plot(trial['y'][:, 0, ch], '--', alpha=0.7, label=f'Target {ch}', linewidth=2)

        # Add performance text
        perf_text = ""
        if 'accuracy' in performance:
            perf_text = f"Accuracy: {performance['accuracy']:.3f}"
            if 'response_start_time' in performance:
                perf_text += f"\nResponse start: {performance['response_start_time']}"
        elif 'mean_timing_error' in performance:
            perf_text = f"Timing Error (steps): {performance['mean_timing_error']:.3f}"
            if 'mean_completion' in performance:
                perf_text += f"\nMean Completion: {performance['mean_completion']:.3f}"
            if 'go_start_time' in performance:
                perf_text += f"\nGo start: {performance['go_start_time']}"

        # NEW: Add NaN warning if detected
        if test_result['has_nan']:
            perf_text += f"\n*** NaN DETECTED ***"
            if test_result['nan_details']['outputs']:
                perf_text += f"\nNaN in outputs: YES"
            if test_result['nan_details']['states']:
                perf_text += f"\nNaN in states: YES"
            if test_result['nan_details']['cost']:
                perf_text += f"\nNaN in cost: YES"

        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat' if not test_result['has_nan'] else 'salmon', alpha=0.8))

        ax.set_title(f'Model Outputs vs Targets - {model_name}\nCondition {condition_idx + 1}: {condition}',
                     fontsize=14)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Output Activity')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_model_outputs.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_hidden_states(self, test_result, model_name, save_dir):
        """Plot hidden state evolution separately"""
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

        # NEW: Add NaN warning if detected
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

    def plot_cardiac_data(self, test_result, model_name, save_dir):
        """Plot cardiac data separately (piezo models only)"""
        cardiac_data = test_result['cardiac_data']
        condition = test_result['condition']
        condition_idx = test_result['condition_idx']

        if cardiac_data is None:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot cardiac pressure and R-peaks
        pressure = cardiac_data['hb_sequence'][:, 1]  # Pressure column
        r_peaks = cardiac_data['hb_sequence'][:, 2]  # R-peaks column
        time_cardiac = np.arange(len(pressure)) / cardiac_data['cardiac_sampling_rate']

        ax1.plot(time_cardiac, pressure, 'b-', label='Cardiac Pressure', alpha=0.8, linewidth=2)

        # Mark R-peaks
        r_peak_times = time_cardiac[r_peaks == 1]
        if len(r_peak_times) > 0:
            ax1.scatter(r_peak_times, pressure[r_peaks == 1],
                        color='red', s=50, label='R-peaks', zorder=5)

        ax1.set_title(f'Cardiac Pressure (HR: {cardiac_data["heart_rate"]:.1f} BPM)')
        ax1.set_ylabel('Cardiac Pressure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot pressure slices (what the piezo actually sees)
        if 'pressure_slices' in cardiac_data:
            slices = cardiac_data['pressure_slices']
            slice_means = [np.mean(s) for s in slices]
            slice_times = np.arange(len(slice_means)) * 20 / 1000  # Convert to seconds

            ax2.plot(slice_times, slice_means, 'g-', label='Slice Means (Piezo Input)', linewidth=2, marker='o')
            ax2.set_title('Piezo Input (Pressure Slice Means)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Mean Pressure per Slice')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Cardiac Data - {model_name}\nCondition {condition_idx + 1}: {condition}', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{model_name}_condition_{condition_idx + 1}_cardiac_data.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_model_results_separate(self, results):
        """Create separate visualizations for each model"""
        model_name = results['model_name']
        rule_name = results['rule_name']
        use_piezo = results['use_piezo']
        test_results = results['test_results']

        if not test_results:
            print(f"No test results to plot for {model_name}")
            return

        # Create model-specific directory
        model_dir = os.path.join(self.base_results_dir, f"{model_name}_{rule_name}_plots")
        tools.mkdir_p(model_dir)

        saved_plots = []

        for test_result in test_results:
            # Plot each aspect separately
            task_plot = self.plot_task_structure(test_result, model_name, model_dir)
            output_plot = self.plot_model_outputs(test_result, model_name, model_dir)
            state_plot = self.plot_hidden_states(test_result, model_name, model_dir)

            saved_plots.extend([task_plot, output_plot, state_plot])

            # Plot cardiac data if available
            if use_piezo and test_result['cardiac_data'] is not None:
                cardiac_plot = self.plot_cardiac_data(test_result, model_name, model_dir)
                if cardiac_plot:
                    saved_plots.append(cardiac_plot)

        print(f"Saved {len(saved_plots)} separate plots to: {model_dir}")
        return saved_plots

    def _discover_model_directories(self):
        """Auto-discover trained model directories - FIXED to find both piezo and non-piezo models"""
        model_dirs = []

        # Define possible base directories for different model types
        possible_base_dirs = [
            # Piezo models
            'enhanced_piezo_comparison_results',
            'enhanced_piezo_production_results',
            'enhanced_piezo_time_bisection_results',
            # Standard models (common naming patterns)
            'comparison_results',
            'production_results',
            'bisection_results',
            'interval_comparison_results',
            'interval_production_results',
            'time_bisection_results',
            'model',  # Default model directory
        ]

        for base_dir in possible_base_dirs:
            if os.path.exists(base_dir):
                # Infer rule name from directory name
                if 'comparison' in base_dir.lower():
                    rule_name = 'interval_comparison'
                elif 'production' in base_dir.lower():
                    rule_name = 'interval_production'
                elif 'bisection' in base_dir.lower():
                    rule_name = 'time_bisection'
                elif base_dir == 'model':
                    # Check subdirectories for rule-specific models
                    for rule_subdir in ['interval_comparison', 'interval_production', 'time_bisection']:
                        rule_path = os.path.join(base_dir, rule_subdir)
                        if os.path.exists(rule_path):
                            self._search_model_subdirs(rule_path, rule_subdir, model_dirs)
                    continue
                else:
                    print(f"Warning: Cannot infer rule name from directory {base_dir}, skipping")
                    continue

                # Search for models in this base directory
                self._search_model_subdirs(base_dir, rule_name, model_dirs)

        return model_dirs

    def _search_model_subdirs(self, base_dir, rule_name, model_dirs):
        """Search for model subdirectories in a base directory"""
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Check if this looks like a run directory
                    if 'run_' in item or item.startswith('w2_') or item.isdigit():
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
        candidates = []

        # Check for finalResult
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
        plt.show()

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

    # if all_results:
    #     comparison_path = os.path.join(self.base_results_dir, "model_comparison.png")
    #     self.compare_model_performance(all_results, comparison_path)
    #
    #     # NEW: Generate per-condition consolidated graphs
    #     self.plot_condition_consolidated_graphs(all_results)
    #
    #     # NEW: Generate interval length scatter plot
    #     self.plot_interval_length_vs_completion(all_results)

    def _has_valid_model(self, model_path):
        """Check if a directory contains a valid model"""
        model_file = os.path.join(model_path, 'model.pth')
        hp_file = os.path.join(model_path, 'hp.json')
        return os.path.exists(model_file) and os.path.exists(hp_file)

    def _generate_summary_report(self, all_results):
        """Generate a text summary report - ENHANCED with NaN tracking"""
        report_path = os.path.join(self.base_results_dir, "comprehensive_test_report.txt")

        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL TEST REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall summary
            f.write(f"Total models tested: {len(all_results)}\n")

            comparison_models = [r for r in all_results if r['rule_name'] == 'interval_comparison']
            production_models = [r for r in all_results if r['rule_name'] == 'interval_production']

            f.write(f"Interval comparison models: {len(comparison_models)}\n")
            f.write(f"Interval production models: {len(production_models)}\n")
            bisection_models = [r for r in all_results if r['rule_name'] == 'time_bisection']
            f.write(f"Time bisection models: {len(bisection_models)}\n")

            piezo_models = [r for r in all_results if r['use_piezo']]
            standard_models = [r for r in all_results if not r['use_piezo']]

            f.write(f"Piezo-enhanced models: {len(piezo_models)}\n")
            f.write(f"Standard models: {len(standard_models)}\n\n")

            # NEW: NaN summary
            f.write("NaN OCCURRENCE SUMMARY\n")
            f.write("-" * 25 + "\n")

            total_nan_conditions = 0
            total_conditions = 0

            for result in all_results:
                nan_count = result['performance_summary'].get('total_nan_conditions', 0)
                condition_count = result['performance_summary'].get('total_conditions', 0)
                total_nan_conditions += nan_count
                total_conditions += condition_count

                if nan_count > 0:
                    nan_rate = nan_count / condition_count if condition_count > 0 else 0
                    f.write(
                        f"  {result['model_name']}: {nan_count}/{condition_count} NaN conditions ({nan_rate:.1%})\n")

            overall_nan_rate = total_nan_conditions / total_conditions if total_conditions > 0 else 0
            f.write(f"  OVERALL: {total_nan_conditions}/{total_conditions} NaN conditions ({overall_nan_rate:.1%})\n\n")

            # Detailed results for each model
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

        print(f"Generated summary report: {report_path}")


def main():
    """Run comprehensive model testing"""
    tester = ComprehensiveModelTester()
    results = tester.run_comprehensive_test()

    print(f"\nComprehensive testing complete!")
    print(f"Results saved to: {tester.base_results_dir}")
    print(f"Total models tested: {len(results)}")


if __name__ == "__main__":
    main()