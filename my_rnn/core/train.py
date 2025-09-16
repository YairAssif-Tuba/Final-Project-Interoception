"""Main training loop - SIMPLIFIED PIEZO VERSION

This module implements the training infrastructure for RNN models on cognitive tasks.
All pretraining logic has been removed since the simplified piezo interface
has no trainable parameters.

CHANGES:
- Removed all cardiac pretraining code
- Removed pretraining checkpoints
- Simplified trainer initialization
- Clean single-phase training only
"""

from __future__ import division

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import time
import sys
import os

import train_stepper
import network
import tools
import dataset



def collate_fn(batch):
    """Handle batches in DataLoader"""
    return batch[0]


def get_perf_prod(output, target_time, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5):
    """Evaluate performance on the interval production task."""
    batch_size = output.shape[1]

    # Detect premature responses during fixation
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i] > fix_strength) > 0 for i in range(batch_size)])

    # Detect absence of response after go cue
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:, i] > action_threshold) == 0 for i in range(batch_size)])

    # Combine failure modes
    fail_action = action_at_fix + no_action_at_motion

    # Find when actions occurred (relative to go cue)
    action_time = np.array([np.argmax(output[fix_end[i]:, i] > action_threshold) for i in range(batch_size)])

    # Calculate relative timing error
    rel_action_time = np.abs(action_time - target_time) / target_time

    # Calculate success probability
    success_action_prob = 1 - np.sum(fail_action) / batch_size

    # Calculate mean timing error for successful trials
    mean_rel_action_time = np.mean(rel_action_time[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_rel_action_time


def get_perf_discrim(output, target_choice, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5,
                     response_duration=int(300 / 20)):
    """Evaluate performance on the interval comparison task."""
    """Evaluate performance on the interval comparison task."""
    dts = output.shape[0]
    batch_size = output.shape[1]
    print("dts:", dts)
    print("batch size:", batch_size)
    print("output channels", output.shape[2])
    batch_size = output.shape[1]

    # Detect premature responses during fixation
    action_at_fix = np.array(
        [np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])

    # Detect absence of response after go cue
    no_action_at_motion = np.array(
        [np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold) == 0 for i in
         range(batch_size)])

    # Detect multiple responses (activating multiple output units)
    both_action_at_motion = np.array(
        [np.sum(np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for
         i in range(batch_size)])

    # Combine failure modes
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion
    print(np.array([
        output[fix_end[i]:fix_end[i] + response_duration, i, :]
        for i in range(batch_size)
    ]))
    print("I WAS SUPPOSED TO PRINT")

    # Find when responses occurred
    action_time = np.array(
        [np.argmax(np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold, axis=1) > 0) for i
         in range(batch_size)])

    # Extract binary responses at action time
    action = np.concatenate(
        [(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int32) for i in range(batch_size)],
        axis=0)

    # Convert activations to choice (-1 or 1)
    choice = action[:, 0] - action[:, 1]

    # Check if choices match targets
    choice_correct = ((choice - target_choice) == 0)

    # Calculate success probability
    success_action_prob = 1 - np.sum(fail_action) / batch_size

    # Calculate mean choice accuracy for successful trials
    mean_choice_correct = np.mean(choice_correct[np.argwhere(1 - fail_action)])

    return success_action_prob, 1 - mean_choice_correct


def get_perf_bisection(output, target_choice, fix_start, fix_end,
                       fix_strength=0.2, action_threshold=0.3, response_duration=int(300 / 20)):
    debug = False
    # dts = output.shape[0]
    batch_size = output.shape[1]
    if debug:
        print([output[fix_end[i]:fix_end[i] + response_duration, i, :] for i in range(batch_size)])
    # Detect premature responses during fixation
    action_at_fix = np.array(
        [np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])

    # Detect absence of response after go cue
    no_action_at_motion = np.array(
        [np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold) == 0 for i in
         range(batch_size)])

    # Detect multiple responses (activating multiple output units)
    both_action_at_motion = np.array(
        [np.sum(np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for
         i in range(batch_size)])

    # Combine failure modes
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion

    # Find when responses occurred
    action_time = np.array(
        [np.argmax(np.sum(output[fix_end[i]:fix_end[i] + response_duration, i, :] > action_threshold, axis=1) > 0) for i
         in range(batch_size)])

    # Extract binary responses at action time
    action = np.concatenate(
        [(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int32) for i in range(batch_size)],
        axis=0)

    # Convert activations to choice (-1 or 1)
    choice = action[:, 0] - action[:, 1]

    # Check if choices match targets
    choice_correct = ((choice - target_choice) == 0)

    # Calculate success probability
    success_action_prob = 1 - np.sum(fail_action) / batch_size
    if debug:
        print("TARGET_CHOICE:\n", target_choice, "\nCHOICE:\n", choice, "\nCHOICE_CORRECT:\n", choice_correct)
    # Calculate mean choice accuracy for successful trials
    valid_mask = (fail_action == 0)
    mean_choice_correct = np.mean(choice_correct[valid_mask])

    return success_action_prob, 1 - mean_choice_correct


def get_perf_gaussian(output, target_choice,
                      action_threshold=0.5, debug=False):
    """
    Evaluate performance assuming output shape is [time, batch_size, channels],
    with time dimension = 1.

    Args:
        output: np array of shape [time=1, batch_size, channels]
        target_choice: np array of shape [batch_size]
        fix_start, fix_end, fix_strength, response_duration: ignored but included for interface compatibility
        action_threshold: float, threshold for action detection

    Returns:
        success_action_prob: proportion of trials without failed actions
        mean_choice_error: mean choice error across valid trials
    """

    batch_size = output.shape[1]

    # Extract the only time step (shape [batch_size, channels])
    output_t = output[0, :, :]
    print(output_t)
    # Determine per batch:
    # (1) if both channels crossed threshold (multiple responses)
    both_action = np.sum(output_t > action_threshold, axis=1) > 1

    # (2) if no channel responded (no action)
    no_action = np.sum(output_t > action_threshold, axis=1) == 0

    # Combine failure modes
    fail_action = both_action | no_action

    # Convert outputs to choices (-1 or 1) if not failed
    choice = (output_t[:, 0] > action_threshold).astype(int) - (output_t[:, 1] > action_threshold).astype(int)

    # Check choice correctness
    choice_correct = (choice == target_choice)

    # Calculate success probability
    success_action_prob = 1 - np.sum(fail_action) / batch_size
    if debug:
        print("TARGET_CHOICE:\n", target_choice, "\nCHOICE:\n", choice, "\nCHOICE_CORRECT:\n", choice_correct)
    # Calculate mean choice accuracy for successful trials
    valid_mask = (fail_action == 0)
    mean_choice_correct = np.mean(choice_correct[valid_mask])

    # Return success probability and choice error (1 - accuracy)
    return success_action_prob, 1 - mean_choice_correct


class Trainer(object):
    """RNN model trainer class - SIMPLIFIED VERSION.

    This class handles the complete training process without pretraining:
    - Loading/initializing the model
    - Setting up data loaders
    - Running the training loop
    - Evaluating model performance
    - Saving model checkpoints

    SIMPLIFIED: No pretraining phase - direct single-phase training only
    """

    def __init__(self, rule_name=None, model=None, hp=None, model_dir=None, is_cuda=False, **kwargs):
        """Initialize the Trainer.

        Args:
            rule_name: Name of the task rule to train on
            model: Pre-built RNN model (if None, will create or load one)
            hp: Hyperparameters (if None, will load from model_dir)
            model_dir: Directory for saving/loading models and logs
            is_cuda: Whether to use GPU acceleration
            **kwargs: Additional arguments passed to TaskDataset and RNN
        """
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir

        self.rule_name = rule_name
        self.is_cuda = is_cuda
        self.use_piezo = hp.get('use_piezo', False) if hp is not None else False

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load or create hyperparameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # Calculate alpha (time scale hyperparameter)
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        hp['rng'] = np.random.RandomState(42)
        self.hp = hp

        # Save hyperparameters if not already saved
        fh_fname = os.path.join(model_dir, 'hp.json')
        if not os.path.isfile(fh_fname):
            tools.save_hp(hp, model_dir)

        # Initialize or load the RNN model
        kwargs['rule_name'] = rule_name
        if model is None:
            self.model = network.RNN(hp, is_cuda, **kwargs)
            self.model.load(model_dir)
        else:
            self.model = model

        # Apply scale-compatible initialization for piezo models
        #if self.use_piezo:
            #self.apply_scale_compatible_initialization()

        # Print model configuration
        if self.use_piezo:
            print("🫀 Simplified piezo interface enabled")
            piezo_info = self.model.get_piezo_info()
            print(f"   Connected receptors: {piezo_info['num_connected']}")
            print(f"   Trainable params: {piezo_info['trainable_params']} (connectivity only)")
        else:
            print("🚫 Standard training without piezo interface")

        # Initialize or load training logs
        self.log = tools.load_log(model_dir)
        if self.log is None:
            self.log = defaultdict(list)
            self.log['model_dir'] = model_dir

        # Create training stepper for handling parameter updates
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda, mode='main_task')

        # Remove rule_name from kwargs to avoid duplication
        del kwargs['rule_name']
        #if rule_name == 'time_bisection':
            #kwargs['training_only'] = False

        # Set up data loaders for training and testing
        dataset_train = dataset.TaskDataset(rule_name, hp, mode='train', is_cuda=is_cuda, **kwargs)
        dataset_test = dataset.TaskDataset(rule_name, hp, mode='test', is_cuda=is_cuda, **kwargs)

        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=1,
                                           collate_fn=collate_fn)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1,
                                          collate_fn=collate_fn)

        # Initialize tracking variables
        self.min_cost = np.inf
        self.model_save_idx = 0

    def apply_scale_compatible_initialization(self):
        """
        Apply scale-compatible initialization for simplified piezo interface.
        Prevents training instability by using conservative parameter scaling.
        """
        if not self.use_piezo:
            return  # Only needed for piezo models

        print("🔧 Applying scale-compatible initialization...")

        # Conservative parameter scaling (same values as original for consistency)
        connectivity_scale = 0.01  # Much smaller than default 0.3
        main_network_scale = 0.05  # Smaller than default 0.3

        with torch.no_grad():
            # Reinitialize connectivity parameter with small scale
            if hasattr(self.model, 'piezo_connectivity') and self.model.piezo_connectivity is not None:
                torch.nn.init.normal_(self.model.piezo_connectivity, mean=0.0, std=connectivity_scale)
                print(f"   Reinitialized connectivity with std={connectivity_scale}")

            # Reinitialize main network with smaller scale to prevent NaN explosions
            torch.nn.init.normal_(self.model.weight_ih, mean=0.0, std=main_network_scale)
            torch.nn.init.normal_(self.model.weight_hh, mean=0.0, std=main_network_scale)
            torch.nn.init.normal_(self.model.weight_out, mean=0.0, std=main_network_scale)

            print(f"   Reinitialized main network with std={main_network_scale}")

        print("✅ Scale-compatible initialization complete")

    def do_eval(self):
        """Evaluate the model on test data and save checkpoints."""
        print('Trial {:7d}'.format(self.log['trials'][-1]) +
              '  | Time {:0.2f} s'.format(self.log['times'][-1]))

        for i_batch, sample_batched in enumerate(self.dataloader_test):
            # Add this debug right here:
            print(f"\nDEBUG TEST DATA TARGETS:")
            test_targets = sample_batched['target_outputs']
            print(f"  Test targets shape: {test_targets.shape}")
            print(f"  Test targets sum: {test_targets.sum()}")
            print(f"  Test non-zero targets: {(test_targets != 0).sum()}")

            # Compare to what we expect from intervals
            if 'prod_interval1' in sample_batched and 'prod_interval2' in sample_batched:
                int1 = sample_batched['prod_interval1'] * self.hp['dt']
                int2 = sample_batched['prod_interval2'] * self.hp['dt']
                expected_targets = (int1 <= int2).sum()  # How many should be [0,1]
                print(f"  Expected [0,1] targets: {expected_targets}")
                print(f"  Expected [1,0] targets: {len(int1) - expected_targets}")
            # Initialize temporary cost storage
            clsq_tmp = list()
            creg_tmp = list()

            # Move data to GPU if using CUDA
            if self.is_cuda:
                sample_batched['inputs'] = sample_batched['inputs'].cuda()
                sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

            sample_batched['rule_name'] = self.rule_name

            # Calculate costs without updating weights
            with torch.no_grad():
                self.train_stepper.cost_fcn(**sample_batched)

            # Store costs
            clsq_tmp.append(self.train_stepper.cost_lsq.detach().cpu().numpy())
            creg_tmp.append(self.train_stepper.cost_reg.detach().cpu().numpy())

            # Add to log
            self.log['cost_'].append(np.mean(clsq_tmp, dtype=np.float64))
            self.log['creg_'].append(np.mean(creg_tmp, dtype=np.float64))

            # Print cost information
            cost_str = '| cost {:0.6f}'.format(np.mean(clsq_tmp)) + '| c_reg {:0.6f}'.format(np.mean(creg_tmp))
            if self.use_piezo:
                cost_str += ' | 🫀 SIMPLE'
            print(cost_str)

            sys.stdout.flush()

            # Save model if cost improves
            if clsq_tmp[-1] < self.min_cost:
                self.min_cost = clsq_tmp[-1]
                print('save model!')
                self.model.save(self.model_dir)

            tools.save_log(self.log)

            # Save model routinely (checkpoint)
            routine_save_path = os.path.join(self.model_dir, str(self.model_save_idx))
            self.model_save_idx = self.model_save_idx + 1
            tools.mkdir_p(routine_save_path)
            self.model.save(routine_save_path)

            # Task-specific performance evaluation
            if self.rule_name == 'interval_production':
                success_action_prob, mean_rel_action_time = get_perf_prod(
                    self.train_stepper.outputs[:, :, 0].detach().cpu().numpy(),
                    sample_batched['prod_interval'],
                    sample_batched['epochs']['stim1'][1],
                    sample_batched['epochs']['go_cue'][1]
                )

                # Convert numpy values to Python types for JSON serialization
                success_action_prob = success_action_prob.tolist()
                mean_rel_action_time = mean_rel_action_time.tolist()

                # Save performance metrics
                self.info = dict()
                self.info['cost'] = clsq_tmp[-1].tolist()
                self.info['creg'] = creg_tmp[-1].tolist()
                self.info['success_action_prob'] = success_action_prob
                self.info['mean_rel_action_time'] = mean_rel_action_time
                self.info['model_dir'] = routine_save_path
                self.info['use_piezo'] = self.use_piezo
                tools.save_log(self.info)
                tools.save_hp(self.hp, routine_save_path)

                # Print performance metrics
                perf_str = '| success_action_prob {:0.6f}'.format(
                    success_action_prob) + '| mean_rel_action_time {:0.6f}'.format(mean_rel_action_time)
                if self.use_piezo:
                    perf_str += ' | 🫀'
                print(perf_str)

            elif self.rule_name == 'interval_comparison':
                success_action_prob, mean_choice_error = get_perf_discrim(
                    self.train_stepper.outputs.detach().cpu().numpy(),
                    2 * (sample_batched['prod_interval1'] > sample_batched['prod_interval2']) - 1,
                    sample_batched['epochs']['stim1'][0] + 5,
                    sample_batched['epochs']['go'][0]
                )

                # Convert numpy values to Python types for JSON serialization
                success_action_prob = success_action_prob.tolist()
                mean_choice_error = mean_choice_error.tolist()

                # Save performance metrics
                self.info = dict()
                self.info['cost'] = clsq_tmp[-1].tolist()
                self.info['creg'] = creg_tmp[-1].tolist()
                self.info['success_action_prob'] = success_action_prob
                self.info['mean_choice_error'] = mean_choice_error
                self.info['model_dir'] = routine_save_path
                self.info['use_piezo'] = self.use_piezo
                tools.save_log(self.info)
                tools.save_hp(self.hp, routine_save_path)

                # Print performance metrics
                perf_str = '| success_action_prob {:0.6f}'.format(
                    success_action_prob) + '| mean_choice_error {:0.6f}'.format(mean_choice_error)
                if self.use_piezo:
                    perf_str += ' | 🫀'
                print(perf_str)



            elif self.rule_name == 'time_bisection':

                success_action_prob, mean_choice_error = get_perf_bisection(

                    self.train_stepper.outputs.detach().cpu().numpy(),

                    2 * (sample_batched['comparison_intervals'] <= (

                            (sample_batched['short_standard'] + sample_batched['long_standard']) / 2

                    )).astype(np.int32) - 1,

                    sample_batched['epochs']['delay'][0],  # Fixation start

                    sample_batched['epochs']['delay'][0]  # Fixation end

                )

                # Convert numpy values to Python types for JSON serialization

                success_action_prob = success_action_prob.tolist()

                mean_choice_error = mean_choice_error.tolist()
                # Save performance metrics
                self.info = dict()

                self.info['cost'] = clsq_tmp[-1].tolist()

                self.info['creg'] = creg_tmp[-1].tolist()

                self.info['success_action_prob'] = success_action_prob

                self.info['mean_choice_error'] = mean_choice_error

                self.info['model_dir'] = routine_save_path

                tools.save_log(self.info)

                tools.save_hp(self.hp, routine_save_path)

                # Print performance metrics

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +

                      '| mean_choice_error {:0.6f}'.format(mean_choice_error))


            elif self.rule_name == 'gaussian_bisection':

                success_action_prob, mean_choice_error = get_perf_gaussian(

                    self.train_stepper.outputs.detach().cpu().numpy(),

                    2 * (sample_batched['comparison_intervals'] <= (

                            (sample_batched['short_standard'] + sample_batched['long_standard']) / 2

                    )).astype(np.int32) - 1)

                # Convert numpy values to Python types for JSON serialization

                success_action_prob = success_action_prob.tolist()

                mean_choice_error = mean_choice_error.tolist()

                # Save performance metrics

                self.info = dict()

                self.info['cost'] = clsq_tmp[-1].tolist()

                self.info['creg'] = creg_tmp[-1].tolist()

                self.info['success_action_prob'] = success_action_prob

                self.info['mean_choice_error'] = mean_choice_error

                self.info['model_dir'] = routine_save_path

                tools.save_log(self.info)

                tools.save_hp(self.hp, routine_save_path)

                # Print performance metrics

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +

                      '| mean_choice_error {:0.6f}'.format(mean_choice_error))
            # Return performance metrics after processing the first batch
            if i_batch == 0:
                if self.rule_name == 'interval_production':
                    return clsq_tmp[-1], success_action_prob, mean_rel_action_time
                elif self.rule_name == 'interval_comparison':
                    return clsq_tmp[-1], success_action_prob, mean_choice_error
                elif self.rule_name == 'time_bisection' or self.rule_name == 'gaussian_bisection':
                    return clsq_tmp[-1], success_action_prob, mean_choice_error

    def diagnose_interval_comparison(self, sample_batched, outputs):
        """Fixed diagnostic that matches the actual evaluation logic"""

        if self.rule_name != 'interval_comparison':
            return

        print(f"\n=== FIXED INTERVAL COMPARISON DIAGNOSIS ===")

        # Get intervals - handle both tensor and numpy cases
        if hasattr(sample_batched['prod_interval1'], 'cpu'):
            interval1 = sample_batched['prod_interval1'].cpu().numpy() * self.hp['dt']
            interval2 = sample_batched['prod_interval2'].cpu().numpy() * self.hp['dt']
        else:
            interval1 = sample_batched['prod_interval1'] * self.hp['dt']
            interval2 = sample_batched['prod_interval2'] * self.hp['dt']

        # USE THE SAME LOGIC AS THE ACTUAL EVALUATION
        # From get_perf_discrim function:
        target_choice = 2 * (interval1 > interval2) - 1  # +1 if I1>I2, -1 if I2>=I1

        # Get final outputs (average of last 20 timesteps)
        final_outputs = outputs[-20:, :, :].mean(axis=0)  # [B, 2]

        # Convert to choice format (same as evaluation)
        # choice = action[:, 0] - action[:, 1]
        network_choice = final_outputs[:, 0] - final_outputs[:, 1]  # Positive = choose I1, negative = choose I2

        # Convert to binary choices for analysis
        network_binary = (network_choice > 0).astype(int)  # 1 if chose I1, 0 if chose I2
        target_binary = (target_choice > 0).astype(int)  # 1 if I1>I2, 0 if I2>=I1

        # Calculate accuracy using the same logic as evaluation
        choice_correct = (network_choice * target_choice > 0)  # Both same sign = correct
        accuracy = choice_correct.mean()

        print(f"Evaluation-matched accuracy: {accuracy:.3f}")
        print(f"Network choice distribution: I1={np.sum(network_binary == 1)}, I2={np.sum(network_binary == 0)}")
        print(f"Correct choice distribution: I1={np.sum(target_binary == 1)}, I2={np.sum(target_binary == 0)}")

        # Check for bias
        network_bias = abs(network_binary.mean() - 0.5)
        if network_bias > 0.3:
            print(
                f"WARNING: Network biased toward {'I1' if network_binary.mean() > 0.5 else 'I2'} (bias: {network_bias:.3f})")

        # Decision confidence
        confidence = np.abs(network_choice)
        weak_decisions = (confidence < 0.1).sum()
        print(
            f"Weak decisions (conf < 0.1): {weak_decisions}/{len(confidence)} ({weak_decisions / len(confidence) * 100:.1f}%)")
        print(f"Mean confidence: {confidence.mean():.3f}")

        # Show examples with the evaluation format
        print(f"\nFirst 8 examples (evaluation format):")
        print(f"{'I1(ms)':>8} {'I2(ms)':>8} {'Target':>6} {'Network':>7} {'Correct':>7} {'Conf':>6}")
        for i in range(min(8, len(interval1))):
            correct = "YES" if choice_correct[i] else "NO"
            target_str = "I1" if target_choice[i] > 0 else "I2"
            network_str = "I1" if network_choice[i] > 0 else "I2"
            print(
                f"{interval1[i]:8.0f} {interval2[i]:8.0f} {target_str:>6} {network_str:>7} {correct:>7} {confidence[i]:6.3f}")

        # Difficulty analysis
        difficulties = np.abs(interval1 - interval2) / np.maximum(interval1, interval2)
        easy_mask = difficulties > 0.2
        hard_mask = difficulties < 0.1

        if easy_mask.sum() > 0:
            easy_acc = choice_correct[easy_mask].mean()
            print(f"Easy trials (>20% diff): {easy_acc:.3f} accuracy")

        if hard_mask.sum() > 0:
            hard_acc = choice_correct[hard_mask].mean()
            print(f"Hard trials (<10% diff): {hard_acc:.3f} accuracy")

        print("=" * 50)
    def save_final_result(self):
        """Save the final model when performance criteria are met."""
        save_path = os.path.join(self.model_dir, 'finalResult')
        tools.mkdir_p(save_path)
        self.model.save(save_path)
        self.info['model_dir'] = save_path
        self.info['use_piezo'] = self.use_piezo
        tools.save_log(self.info)
        tools.save_hp(self.hp, save_path)

    def train(self, max_samples=1e7, display_step=500, max_model_save_idx=150):
        """Train the network - SIMPLIFIED VERSION."""

        # Display hyperparameters
        print("\n" + "=" * 50)
        print("🚀 SIMPLIFIED PIEZO TRAINING")
        print("=" * 50)

        for key, val in self.hp.items():
            if key != 'rng':  # Skip rng object
                print('{:20s} = '.format(key) + str(val))

        if self.use_piezo:
            print("\n🫀 SIMPLIFIED PIEZO: ACTIVE")
            print("   No pretraining required")
            print("   Only connectivity parameter trainable")
        else:
            print("\n🚫 PIEZO: DISABLED")
        # List for performance graph
        self.error_over_trials = []
        self.trial_counts = []
        # Display hyperparameters
        for key, val in self.hp.items():
            print('{:20s} = '.format(key) + str(val))


        # Record time
        t_start = time.time()
        import task
        print(
            f"DEBUG: Before training loop, rule_mapping function: {task.rule_mapping['interval_comparison'].__name__}")
        print(f"DEBUG: Before training loop, function id: {id(task.rule_mapping['interval_comparison'])}")

        # Main training loop
        for step, sample_batched in enumerate(self.dataloader_train):
            #print(f"DEBUG: Training step {step}, got sample_batched")

            try:
                interrupt = False

                # Move data to GPU if CUDA is enabled
                if self.is_cuda:
                    sample_batched['inputs'] = sample_batched['inputs'].cuda()
                    sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                    sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                    sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                    sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

                sample_batched['rule_name'] = self.rule_name

                # Set L2 regularization on firing rate
                if self.model_save_idx < 5:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(1e-3, device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(1e-3, device=self.device)
                else:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(self.hp['l2_firing_rate'],
                                                                         device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(self.hp['l2_firing_rate'], device=self.device)

                # Perform training step
                self.train_stepper.stepper(**sample_batched)
                if step % 1000 == 0:  # Every 1000 steps
                    print(f"\nDEBUG TRAINING TARGETS AT STEP {step}:")
                    targets = sample_batched['target_outputs']
                    print(f"  Targets shape: {targets.shape}")
                    print(f"  Targets sum: {targets.sum()}")
                    print(f"  Non-zero targets: {(targets != 0).sum()}")
                    if hasattr(targets, 'max'):
                        print(f"  Targets range: [{targets.min():.3f}, {targets.max():.3f}]")

                    # Check a few sample targets
                    if targets.sum() > 0:
                        print(f"  Sample non-zero targets found!")
                    else:
                        print(f"  ERROR: All targets are zero - network learning nothing!")
                # Print network output every 64000 trials
                current_trial = step * self.hp['batch_size_train']
                if current_trial > 0 and current_trial % 64000 == 0:
                    print(f"\n=== NETWORK OUTPUT ANALYSIS AT TRIAL {current_trial} ===")

                    # Get the outputs from the last forward pass
                    outputs = self.train_stepper.outputs.detach().cpu().numpy()
                    print(f"Output shape: {outputs.shape}")
                    print(f"Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                    print(f"Output mean: {outputs.mean():.6f}, std: {outputs.std():.6f}")

                    # Print sample outputs for first batch element
                    if outputs.shape[1] > 0:  # Check if we have batch elements
                        print(f"Sample outputs (first batch, last 10 timesteps):")
                        sample_outputs = outputs[-10:, 0, :]  # Last 10 timesteps, first batch
                        for t, out in enumerate(sample_outputs):
                            print(f"  t-{9 - t}: {out}")

                    # Check for output saturation/collapse
                    output_near_zero = (np.abs(outputs) < 0.001).mean()
                    output_saturated = (np.abs(outputs) > 0.999).mean()
                    print(f"Fraction near zero (<0.001): {output_near_zero:.4f}")
                    print(f"Fraction saturated (>0.999): {output_saturated:.4f}")

                    # Check hidden states too
                    if hasattr(self.train_stepper, 'state_binder') and self.train_stepper.state_binder is not None:
                        states = self.train_stepper.state_binder.detach().cpu().numpy()
                        print(f"Hidden states - range: [{states.min():.6f}, {states.max():.6f}]")
                        print(f"Hidden states - mean: {states.mean():.6f}, std: {states.std():.6f}")

                        # Check for NaN or infinite values
                        if np.isnan(states).any():
                            print("WARNING: NaN detected in hidden states!")
                        if np.isinf(states).any():
                            print("WARNING: Infinite values detected in hidden states!")

                    self.diagnose_interval_comparison(sample_batched, self.train_stepper.outputs.detach().cpu().numpy())

                    print("=" * 60 + "\n")

                # Periodically evaluate and report performance
                if step % display_step == 0:
                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)

                    # Evaluate different tasks with their specific criteria
                    if self.rule_name == 'interval_production':
                        cost, success_action_prob, mean_rel_action_time = self.do_eval()
                        if not np.isfinite(cost):
                            return 'error'
                        if success_action_prob > 0.95 and mean_rel_action_time < 0.025:
                            self.save_final_result()
                            break
                        if self.model_save_idx > max_model_save_idx:
                            break


                    elif self.rule_name == 'interval_comparison' or self.rule_name == 'time_bisection' or self.rule_name == 'gaussian_bisection':

                        cost, success_action_prob, mean_rel_action_time = self.do_eval()

                        self.error_over_trials.append(mean_rel_action_time)

                        if not np.isfinite(cost):
                            return 'error', self.error_over_trials, self.trial_counts

                        if success_action_prob > 0.95 and mean_rel_action_time < 0.02:
                            self.save_final_result()

                            break

                        if self.model_save_idx > max_model_save_idx and self.rule_name == 'interval_comparison':
                            break

                        # Now we write the conditions to analyze TB

                        if step * self.hp['batch_size_train'] >= 200000 and (
                                self.rule_name == 'time_bisection' or self.rule_name == 'gaussian_bisection'):
                            self.save_final_result()

                            print("Final results saved")

                            break

                # Break if maximum sample count is reached
                if step * self.hp['batch_size_train'] > max_samples:
                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    self.do_eval()
                    break

            except KeyboardInterrupt:
                interrupt=True
                print("Optimization interrupted by user")
                break

        if self.use_piezo:
            print("🫀 Simplified piezo optimization finished!")
        else:
            print("Optimization finished!")
        if self.rule_name == 'time_bisection' and not interrupt:
            return 'OK'
        else:
            return 'OK'