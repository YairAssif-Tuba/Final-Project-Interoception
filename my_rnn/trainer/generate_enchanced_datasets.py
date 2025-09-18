"""
Enhanced Interval Dataset Generator - SIMPLIFIED
================================================

Generates structured datasets matching the exact enhanced parameter distributions
used in the training scripts, storing only the trial parameters (no batch structure).
"""

import numpy as np
import json
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import PATHS


class EnhancedIntervalDatasetGenerator:
    """Generate datasets matching enhanced training script parameters."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

        # Enhanced parameters (matching training scripts)
        self.enhanced_intervals = {
            'min_interval': 1200,
            'max_interval': 2400,
            'delay_interval': 1000
        }

        # Time bisection parameters (no enhancement override)
        self.bisection_parameters = {
            'short_standard': 1000,
            'long_standard': 2000,
            'std': 40,
            'response_duration': 300
        }

    def generate_interval_comparison_dataset(self, num_batches=4000, batch_size=64, test_split=0.3):
        """Generate dataset for enhanced interval comparison task."""
        print(f"Generating {num_batches} batches of interval comparison data...")

        total_samples = num_batches * batch_size
        train_samples = int(total_samples * (1 - test_split))
        test_samples = total_samples - train_samples

        # Generate all trials
        all_trials = []
        for i in range(total_samples):
            prod_interval1 = self.rng.uniform(
                self.enhanced_intervals['min_interval'],
                self.enhanced_intervals['max_interval']
            )
            prod_interval2 = self.rng.uniform(
                self.enhanced_intervals['min_interval'],
                self.enhanced_intervals['max_interval']
            )
            dly_interval = self.enhanced_intervals['delay_interval']  # Fixed

            # Calculate correct choice (matches original logic)
            correct_choice = 1 if prod_interval1 <= prod_interval2 else 0

            all_trials.append({
                'prod_interval1': float(prod_interval1),
                'prod_interval2': float(prod_interval2),
                'dly_interval': float(dly_interval),
                'correct_choice': int(correct_choice),
                'difficulty': float(abs(prod_interval1 - prod_interval2) / max(prod_interval1, prod_interval2))
            })

        # Split into train/test
        train_trials = all_trials[:train_samples]
        test_trials = all_trials[train_samples:]

        return {
            'train': train_trials,
            'test': test_trials,
            'metadata': {
                'task': 'interval_comparison',
                'total_samples': total_samples,
                'train_samples': train_samples,
                'test_samples': test_samples,
                'batch_size': batch_size,
                'enhanced_intervals': self.enhanced_intervals
            }
        }

    def generate_interval_production_dataset(self, num_batches=4000, batch_size=64, test_split=0.3):
        """Generate dataset for enhanced interval production task."""
        print(f"Generating {num_batches} batches of interval production data...")

        total_samples = num_batches * batch_size
        train_samples = int(total_samples * (1 - test_split))
        test_samples = total_samples - train_samples

        # Generate all trials
        all_trials = []
        for i in range(total_samples):
            prod_interval = self.rng.uniform(
                self.enhanced_intervals['min_interval'],
                self.enhanced_intervals['max_interval']
            )
            # Variable delay: delay_int to delay_int + 400
            dly_interval = self.rng.uniform(
                self.enhanced_intervals['delay_interval'],
                self.enhanced_intervals['delay_interval'] + 400
            )

            all_trials.append({
                'prod_interval': float(prod_interval),
                'dly_interval': float(dly_interval),
                'target_duration': float(prod_interval)
            })

        # Split into train/test
        train_trials = all_trials[:train_samples]
        test_trials = all_trials[train_samples:]

        return {
            'train': train_trials,
            'test': test_trials,
            'metadata': {
                'task': 'interval_production',
                'total_samples': total_samples,
                'train_samples': train_samples,
                'test_samples': test_samples,
                'batch_size': batch_size,
                'enhanced_intervals': self.enhanced_intervals,
                'delay_range': [1000, 1400]
            }
        }

    def generate_time_bisection_dataset(self, num_batches=4000, batch_size=64, test_split=0.3):
        """Generate dataset for time bisection task."""
        print(f"Generating {num_batches} batches of time bisection data...")

        total_samples = num_batches * batch_size
        train_samples = int(total_samples * (1 - test_split))
        test_samples = total_samples - train_samples

        # Generate all trials
        all_trials = []
        for i in range(total_samples):
            # Choose standard (300ms or 900ms)
            chosen_standard = self.rng.choice([
                self.bisection_parameters['short_standard'],
                self.bisection_parameters['long_standard']
            ])

            # Add Gaussian noise
            duration = self.rng.normal(
                chosen_standard,
                self.bisection_parameters['std']
            )
            duration = max(0, duration)  # No negative durations

            # Calculate correct choice (duration <= 600 is "short")
            boundary = (self.bisection_parameters['short_standard'] +
                       self.bisection_parameters['long_standard']) / 2
            correct_choice = 0 if duration <= boundary else 1

            all_trials.append({
                'short_standard': int(self.bisection_parameters['short_standard']),
                'long_standard': int(self.bisection_parameters['long_standard']),
                'std': int(self.bisection_parameters['std']),
                'duration': float(duration),
                'response_duration': int(self.bisection_parameters['response_duration']),
                'correct_choice': int(correct_choice),
                'distance_from_boundary': float(abs(duration - boundary)),
                'chosen_standard': int(chosen_standard)
            })

        # Split into train/test
        train_trials = all_trials[:train_samples]
        test_trials = all_trials[train_samples:]

        return {
            'train': train_trials,
            'test': test_trials,
            'metadata': {
                'task': 'time_bisection',
                'total_samples': total_samples,
                'train_samples': train_samples,
                'test_samples': test_samples,
                'batch_size': batch_size,
                'bisection_parameters': self.bisection_parameters,
                'boundary': 600
            }
        }


def generate_all_enhanced_datasets(output_dir=None,
                                 num_batches=4000,
                                 batch_size=64,
                                 test_split=0.3,
                                 seed=42):
    """Generate all enhanced interval timing datasets."""
    
    # Use configuration path if not specified
    if output_dir is None:
        output_dir = PATHS["ENHANCED_DATASETS_DIR"]

    os.makedirs(output_dir, exist_ok=True)
    generator = EnhancedIntervalDatasetGenerator(seed=seed)

    tasks = {
        #'interval_comparison': generator.generate_interval_comparison_dataset,
        #'interval_production': generator.generate_interval_production_dataset,
        'time_bisection': generator.generate_time_bisection_dataset
    }

    print(f"Generating datasets with {num_batches} batches per task (batch_size={batch_size})")
    print(f"Total samples per task: {num_batches * batch_size:,}")
    print(f"Train/test split: {(1-test_split)*100:.0f}%/{test_split*100:.0f}%\n")

    for task_name, generate_func in tasks.items():
        print(f"Generating {task_name} dataset...")

        # Generate dataset
        dataset = generate_func(
            num_batches=num_batches,
            batch_size=batch_size,
            test_split=test_split
        )

        # Save dataset
        dataset_path = os.path.join(output_dir, f"{task_name}_dataset.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        # Print summary
        print(f"  Saved to: {dataset_path}")
        print(f"  Train samples: {dataset['metadata']['train_samples']:,}")
        print(f"  Test samples: {dataset['metadata']['test_samples']:,}")

        # Print choice balance for classification tasks
        if task_name in ['interval_comparison', 'time_bisection']:
            train_choices = [trial['correct_choice'] for trial in dataset['train']]
            choice_0_pct = train_choices.count(0) / len(train_choices) * 100
            choice_1_pct = train_choices.count(1) / len(train_choices) * 100
            print(f"  Train choice balance: {choice_0_pct:.1f}% / {choice_1_pct:.1f}%")
        print()

    print(f"All enhanced datasets generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total samples across all tasks: {len(tasks) * num_batches * batch_size:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate enhanced interval timing datasets')
    parser.add_argument('--output-dir', default='enhanced_interval_datasets',
                        help='Output directory for datasets')
    parser.add_argument('--num-batches', type=int, default=4000,
                        help='Number of batches per task (default: 4000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--test-split', type=float, default=0.3,
                        help='Fraction of data for testing (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    generate_all_enhanced_datasets(
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        test_split=args.test_split,
        seed=args.seed
    )