"""Task plotting and analysis module.

This module provides comprehensive analysis and plotting capabilities for trained RNN models
on interval timing tasks. It supports various analysis types including:
- compare_tau: Compare performance across different membrane time constants
- compare_intervals: Compare performance across different interval standards
- compare_std: Compare performance across different standard deviations
- compare_reg: Compare performance across different regularization strengths
- learning_curves: Plot learning curves for multiple models
- psychometric: Generate psychometric curves for classification tasks

The module integrates with the existing codebase and follows the established patterns
for model training, evaluation, and result management.
"""

import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from typing import List, Tuple, Dict, Optional, Union

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import train
import default
import tools
import run


class TaskPlotter:
    """Main class for task analysis and plotting.

    This class provides methods for training multiple models with different parameters
    and generating comparative plots and analyses.
    """

    def __init__(self, rule_name: str, base_model_dir: str = None, is_cuda: bool = True):
        """Initialize the TaskPlotter.

        Args:
            rule_name: Name of the task rule to analyze
            base_model_dir: Base directory for saving models (if None, uses current directory)
            is_cuda: Whether to use GPU acceleration
        """
        self.rule_name = rule_name
        self.is_cuda = is_cuda

        if base_model_dir is None:
            self.base_model_dir = os.path.join(os.getcwd(), f"{rule_name}_analysis")
        else:
            self.base_model_dir = base_model_dir

        tools.mkdir_p(self.base_model_dir)

        # Create plots directory
        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Tasks_plots')
        tools.mkdir_p(self.plots_dir)

        # Default hyperparameters
        self.base_hp = default.get_default_hp(rule_name)

        # Analysis results storage
        self.results = defaultdict(list)

    def train_multiple_models(self, param_name: str, param_values: List,
                              n_models: int = 5, **kwargs) -> Dict:
        """Train multiple models with different parameter values.

        Args:
            param_name: Name of the parameter to vary ('tau', 'short_standard', 'long_standard', 'std', etc.)
            param_values: List of parameter values to test
            n_models: Number of models to train for each parameter value
            **kwargs: Additional parameters passed to the trainer

        Returns:
            Dictionary containing training results for each parameter value
        """
        results = defaultdict(list)

        for param_val in param_values:
            print(f"\n=== Training models with {param_name} = {param_val} ===")

            param_errors = []
            param_trial_counts = []

            for i in range(n_models):
                print(f"Training model {i + 1}/{n_models}")

                # Create model directory
                model_dir = os.path.join(self.base_model_dir, f"{param_name}_{param_val}_model_{i}")
                tools.mkdir_p(model_dir)

                # Set up hyperparameters
                hp = self.base_hp.copy()
                hp[param_name] = param_val

                # Add task-specific parameters
                for key, value in kwargs.items():
                    hp[key] = value

                try:
                    # Create trainer and train
                    trainer = train.Trainer(
                        rule_name=self.rule_name,
                        model_dir=model_dir,
                        hp=hp,
                        is_cuda=self.is_cuda,
                        **kwargs
                    )

                    # Train with appropriate display step
                    display_step = math.ceil(5000 / hp['batch_size_train'])
                    status, errors, trial_counts = trainer.train(
                        max_samples=1e7,
                        display_step=display_step
                    )

                    if status == 'OK' and errors and trial_counts:
                        param_errors.append(errors)
                        param_trial_counts.append(trial_counts)
                    else:
                        print(f"Training failed for model {i + 1}")

                except Exception as e:
                    print(f"Error training model {i + 1}: {e}")
                    continue

            if param_errors:
                # Align lengths and compute statistics
                min_len = min(len(e) for e in param_errors)
                errors_trimmed = np.array([e[:min_len] for e in param_errors])
                trial_counts_trimmed = param_trial_counts[0][:min_len]

                mean_errors = np.mean(errors_trimmed, axis=0)
                std_errors = np.std(errors_trimmed, axis=0)

                results[param_val] = {
                    'mean_errors': mean_errors,
                    'std_errors': std_errors,
                    'trial_counts': trial_counts_trimmed,
                    'n_models': len(param_errors)
                }
            else:
                print(f"No successful models for {param_name} = {param_val}")

        return results

    def moving_average(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Compute moving average of data.

        Args:
            data: Input data array
            window_size: Size of the moving average window

        Returns:
            Smoothed data array
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def plot_learning_curves(self, results: Dict, param_name: str,
                             window_size: int = 5, save_plot: bool = True) -> str:
        """Plot learning curves for different parameter values.

        Args:
            results: Results dictionary from train_multiple_models
            param_name: Name of the parameter that was varied
            window_size: Window size for moving average smoothing
            save_plot: Whether to save the plot

        Returns:
            Path to the saved plot file
        """
        plt.figure(figsize=(12, 8))

        for param_val, result in results.items():
            if 'mean_errors' in result:
                # Apply moving average smoothing
                smoothed_errors = self.moving_average(result['mean_errors'], window_size)
                trials_smoothed = result['trial_counts'][len(result['trial_counts']) - len(smoothed_errors):]

                # Plot mean with standard error
                plt.plot(trials_smoothed, smoothed_errors,
                         label=f'{param_name} = {param_val}', linewidth=2)

                # Add confidence interval
                smoothed_std = self.moving_average(result['std_errors'], window_size)
                plt.fill_between(trials_smoothed,
                                 smoothed_errors - smoothed_std,
                                 smoothed_errors + smoothed_std,
                                 alpha=0.3)

        plt.xlabel('Training Trials', fontsize=12)
        plt.ylabel('Mean Error', fontsize=12)
        plt.title(f'{self.rule_name.replace("_", " ").title()} Learning Curves\n'
                  f'Comparing {param_name.replace("_", " ").title()}', fontsize=14)
        plt.legend(title=f'{param_name.replace("_", " ").title()}', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            plot_filename = f"{self.rule_name}_compare_{param_name}.png"
            plot_path = os.path.join(self.plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
            return plot_path
        else:
            plt.show()
            return ""

    def compare_tau(self, tau_values: List[int] = None, n_models: int = 5, **kwargs) -> str:
        """Compare performance across different membrane time constants.

        Args:
            tau_values: List of tau values to test (default: [20, 40, 60, 80, 100])
            n_models: Number of models to train for each tau value
            **kwargs: Additional parameters passed to the trainer

        Returns:
            Path to the saved plot file
        """
        if tau_values is None:
            tau_values = [20, 40, 60, 80, 100]

        print(f"Comparing tau values: {tau_values}")
        results = self.train_multiple_models('tau', tau_values, n_models, **kwargs)

        return self.plot_learning_curves(results, 'tau')

    def compare_intervals(self, short_standards: List[int] = None,
                          long_standards: List[int] = None, n_models: int = 5, **kwargs) -> str:
        """Compare performance across different interval standards.

        Args:
            short_standards: List of short standard values (default: [200, 300, 400])
            long_standards: List of long standard values (default: [800, 900, 1000])
            n_models: Number of models to train for each combination
            **kwargs: Additional parameters passed to the trainer

        Returns:
            Path to the saved plot file
        """
        if short_standards is None:
            short_standards = [200, 300, 400]
        if long_standards is None:
            long_standards = [800, 900, 1000]

        # Create parameter combinations
        param_combinations = []
        for short in short_standards:
            for long in long_standards:
                if short < long:
                    param_combinations.append((short, long))

        print(f"Comparing interval combinations: {param_combinations}")

        # Train models for each combination
        results = {}
        for short, long in param_combinations:
            param_name = f"intervals_{short}_{long}"
            print(f"\n=== Training models with short_standard={short}, long_standard={long} ===")

            param_errors = []
            param_trial_counts = []

            for i in range(n_models):
                print(f"Training model {i + 1}/{n_models}")

                model_dir = os.path.join(self.base_model_dir, f"{param_name}_model_{i}")
                tools.mkdir_p(model_dir)

                hp = self.base_hp.copy()
                hp['short_standard'] = short
                hp['long_standard'] = long

                for key, value in kwargs.items():
                    hp[key] = value

                try:
                    trainer = train.Trainer(
                        rule_name=self.rule_name,
                        model_dir=model_dir,
                        hp=hp,
                        is_cuda=self.is_cuda,
                        short_standard=short,
                        long_standard=long,
                        **kwargs
                    )

                    display_step = math.ceil(5000 / hp['batch_size_train'])
                    status, errors, trial_counts = trainer.train(
                        max_samples=1e7,
                        display_step=display_step
                    )

                    if status == 'OK' and errors and trial_counts:
                        param_errors.append(errors)
                        param_trial_counts.append(trial_counts)

                except Exception as e:
                    print(f"Error training model {i + 1}: {e}")
                    continue

            if param_errors:
                min_len = min(len(e) for e in param_errors)
                errors_trimmed = np.array([e[:min_len] for e in param_errors])
                trial_counts_trimmed = param_trial_counts[0][:min_len]

                results[f"{short}-{long}ms"] = {
                    'mean_errors': np.mean(errors_trimmed, axis=0),
                    'std_errors': np.std(errors_trimmed, axis=0),
                    'trial_counts': trial_counts_trimmed,
                    'n_models': len(param_errors)
                }

        return self.plot_learning_curves(results, 'intervals')

    def compare_std(self, std_values: List[int] = None, n_models: int = 5, **kwargs) -> str:
        """Compare performance across different standard deviations.

        Args:
            std_values: List of standard deviation values (default: [10, 20, 30, 40, 50])
            n_models: Number of models to train for each std value
            **kwargs: Additional parameters passed to the trainer

        Returns:
            Path to the saved plot file
        """
        if std_values is None:
            std_values = [10, 20, 30, 40, 50]

        print(f"Comparing std values: {std_values}")
        results = self.train_multiple_models('std', std_values, n_models, **kwargs)

        return self.plot_learning_curves(results, 'std')

    def compare_reg(self, w2_values: List[float] = None, r2_values: List[float] = None,
                    n_models: int = 5, **kwargs) -> str:
        """Compare performance across different regularization strengths.

        Args:
            w2_values: List of weight regularization values (default: [0.0001, 0.001, 0.01])
            r2_values: List of rate regularization values (default: [0.0001, 0.001, 0.01])
            n_models: Number of models to train for each combination
            **kwargs: Additional parameters passed to the trainer

        Returns:
            Path to the saved plot file
        """
        if w2_values is None:
            w2_values = [0.0001, 0.001, 0.01]
        if r2_values is None:
            r2_values = [0.0001, 0.001, 0.01]

        # Create parameter combinations
        param_combinations = []
        for w2 in w2_values:
            for r2 in r2_values:
                param_combinations.append((w2, r2))

        print(f"Comparing regularization combinations: {param_combinations}")

        results = {}
        for w2, r2 in param_combinations:
            param_name = f"reg_w2_{w2}_r2_{r2}"
            print(f"\n=== Training models with w2={w2}, r2={r2} ===")

            param_errors = []
            param_trial_counts = []

            for i in range(n_models):
                print(f"Training model {i + 1}/{n_models}")

                model_dir = os.path.join(self.base_model_dir, f"{param_name}_model_{i}")
                tools.mkdir_p(model_dir)

                hp = self.base_hp.copy()
                hp['l2_weight'] = w2
                hp['l2_firing_rate'] = r2

                for key, value in kwargs.items():
                    hp[key] = value

                try:
                    trainer = train.Trainer(
                        rule_name=self.rule_name,
                        model_dir=model_dir,
                        hp=hp,
                        is_cuda=self.is_cuda,
                        **kwargs
                    )

                    display_step = math.ceil(5000 / hp['batch_size_train'])
                    status, errors, trial_counts = trainer.train(
                        max_samples=1e7,
                        display_step=display_step
                    )

                    if status == 'OK' and errors and trial_counts:
                        param_errors.append(errors)
                        param_trial_counts.append(trial_counts)

                except Exception as e:
                    print(f"Error training model {i + 1}: {e}")
                    continue

            if param_errors:
                min_len = min(len(e) for e in param_errors)
                errors_trimmed = np.array([e[:min_len] for e in param_errors])
                trial_counts_trimmed = param_trial_counts[0][:min_len]

                results[f"w2={w2}, r2={r2}"] = {
                    'mean_errors': np.mean(errors_trimmed, axis=0),
                    'std_errors': np.std(errors_trimmed, axis=0),
                    'trial_counts': trial_counts_trimmed,
                    'n_models': len(param_errors)
                }

        return self.plot_learning_curves(results, 'regularization')

    def generate_psychometric_curve(self, model_dir: str, test_intervals: List[int] = None,
                                    n_trials: int = 100, **kwargs) -> str:
        """Generate psychometric curve for classification tasks.

        Args:
            model_dir: Directory containing the trained model
            test_intervals: List of intervals to test (default: range from 200 to 1000 ms)
            n_trials: Number of trials per interval
            **kwargs: Additional parameters passed to the runner

        Returns:
            Path to the saved plot file
        """
        if test_intervals is None:
            test_intervals = list(range(200, 1001, 50))

        # Load the trained model
        runner = run.Runner(
            rule_name=self.rule_name,
            model_dir=model_dir,
            is_cuda=self.is_cuda,
            noise_on=False,  # No noise for testing
            **kwargs
        )

        # Test each interval
        proportions_long = []
        for interval in test_intervals:
            correct_long = 0

            for _ in range(n_trials):
                try:
                    # Run the model on this interval
                    if self.rule_name == 'time_bisection':
                        trial, train_stepper = runner.run(
                            short_standard=kwargs.get('short_standard', 300),
                            long_standard=kwargs.get('long_standard', 900)
                        )
                    elif self.rule_name == 'gaussian_bisection':
                        trial, train_stepper = runner.run(
                            short_standard=kwargs.get('short_standard', 300),
                            long_standard=kwargs.get('long_standard', 900)
                        )
                    else:
                        print(f"Psychometric curves not supported for {self.rule_name}")
                        return ""

                    # Get model output
                    output = train_stepper.outputs.detach().cpu().numpy()

                    # Determine choice (assuming output[0] > output[1] means "long")
                    if output.shape[0] == 1:  # Single time step
                        choice = 1 if output[0, 0, 0] > output[0, 0, 1] else -1
                    else:  # Multiple time steps
                        choice = 1 if np.mean(output[:, 0, 0]) > np.mean(output[:, 0, 1]) else -1

                    # Check if choice is correct for this interval
                    midpoint = (kwargs.get('short_standard', 300) + kwargs.get('long_standard', 900)) / 2
                    correct_choice = 1 if interval > midpoint else -1

                    if choice == correct_choice:
                        correct_long += 1

                except Exception as e:
                    print(f"Error testing interval {interval}: {e}")
                    continue

            proportions_long.append(correct_long / n_trials)

        # Plot psychometric curve
        plt.figure(figsize=(10, 6))
        plt.plot(test_intervals, proportions_long, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
        plt.axvline(x=(kwargs.get('short_standard', 300) + kwargs.get('long_standard', 900)) / 2,
                    color='g', linestyle='--', alpha=0.7, label='Midpoint')

        plt.xlabel('Test Interval (ms)', fontsize=12)
        plt.ylabel('Proportion "Long" Responses', fontsize=12)
        plt.title(f'{self.rule_name.replace("_", " ").title()} Psychometric Curve\n'
                  f'Model: {os.path.basename(model_dir)}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = f"{self.rule_name}_psychometric_{os.path.basename(model_dir)}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Psychometric curve saved to: {plot_path}")

        return plot_path


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Task plotting and analysis for RNN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare tau values for time_bisection task
  python task_plotting.py time_bisection compare_tau --n_models 3

  # Compare intervals for time_bisection task
  python task_plotting.py time_bisection compare_intervals --n_models 5

  # Compare regularization for gaussian_bisection task
  python task_plotting.py gaussian_bisection compare_reg --n_models 3

  # Generate psychometric curve for a trained model
  python task_plotting.py time_bisection psychometric --model_dir path/to/model
        """
    )

    parser.add_argument('rule_name',
                        choices=['interval_production', 'interval_comparison', 'time_bisection', 'gaussian_bisection'],
                        help='Name of the task rule to analyze')

    parser.add_argument('analysis_type',
                        choices=['compare_tau', 'compare_intervals', 'compare_std', 'compare_reg', 'psychometric'],
                        help='Type of analysis to perform')

    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of models to train for each parameter value (default: 5)')

    parser.add_argument('--model_dir', type=str,
                        help='Path to trained model directory (required for psychometric analysis)')

    parser.add_argument('--base_model_dir', type=str,
                        help='Base directory for saving models (default: current directory)')

    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')

    parser.add_argument('--short_standard', type=int, default=300,
                        help='Short standard duration in ms (default: 300)')

    parser.add_argument('--long_standard', type=int, default=900,
                        help='Long standard duration in ms (default: 900)')

    parser.add_argument('--std', type=int, default=20,
                        help='Standard deviation for gaussian tasks (default: 20)')

    args = parser.parse_args()

    # Initialize plotter
    is_cuda = not args.cpu
    plotter = TaskPlotter(args.rule_name, args.base_model_dir, is_cuda)

    # Prepare kwargs for task-specific parameters
    kwargs = {
        'short_standard': args.short_standard,
        'long_standard': args.long_standard,
        'std': args.std
    }

    # Perform analysis based on type
    if args.analysis_type == 'compare_tau':
        plot_path = plotter.compare_tau(n_models=args.n_models, **kwargs)
        print(f"Tau comparison completed. Plot saved to: {plot_path}")

    elif args.analysis_type == 'compare_intervals':
        plot_path = plotter.compare_intervals(n_models=args.n_models, **kwargs)
        print(f"Interval comparison completed. Plot saved to: {plot_path}")

    elif args.analysis_type == 'compare_std':
        plot_path = plotter.compare_std(n_models=args.n_models, **kwargs)
        print(f"Standard deviation comparison completed. Plot saved to: {plot_path}")

    elif args.analysis_type == 'compare_reg':
        plot_path = plotter.compare_reg(n_models=args.n_models, **kwargs)
        print(f"Regularization comparison completed. Plot saved to: {plot_path}")

    elif args.analysis_type == 'psychometric':
        if not args.model_dir:
            print("Error: --model_dir is required for psychometric analysis")
            sys.exit(1)

        plot_path = plotter.generate_psychometric_curve(args.model_dir, **kwargs)
        print(f"Psychometric curve generated. Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
