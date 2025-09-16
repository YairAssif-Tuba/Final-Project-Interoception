"""
Training Stepper for Temporal Processing Tasks - SIMPLIFIED VERSION

This module provides a training framework for the RNN models with
simplified piezo interface (no pretraining required).

CHANGES:
- Removed all pretraining phase management
- Simplified parameter handling
- Single-phase training only
- Removed complex phase switching logic
"""

import torch
import numpy as np


class TrainStepper:
    """
    Training stepper for RNN models - SIMPLIFIED VERSION.

    This class manages the optimization process, loss calculation, and regularization
    for training RNN models on temporal tasks.

    SIMPLIFIED: No pretraining phase management needed
    """

    def __init__(self, model, hp, is_cuda=False, mode='main_task'):
        """
        Initialize the training stepper.

        Args:
            model: The RNN model to be trained
            hp: Dictionary containing hyperparameters
            is_cuda: Whether to use GPU acceleration
            mode: Training mode (simplified - always 'main_task')
        """
        # Set up device configuration (GPU/CPU)
        if is_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.step_count = 0

        # Store frequently used constants as tensors for performance
        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)
        self._01 = torch.tensor(0.1, device=self.device)
        self._001 = torch.tensor(0.01, device=self.device)
        self._2 = torch.tensor(2., device=self.device)

        # Store hyperparameters and model
        self.hp = hp
        self.model = model
        self.use_piezo = hp.get('use_piezo', False)
        self.mode = mode

        # Store alpha as tensor
        self.alpha = torch.tensor(hp['alpha'], device=self.device)

        # Set up the RNN model
        self.rnn_net = model
        if is_cuda and torch.cuda.is_available():
            self.rnn_net.to(self.device)

        # Set up references to model weights for regularization
        self.weight_list = [self.rnn_net.weight_ih, self.rnn_net.weight_hh, self.rnn_net.weight_out]
        self.out_weight = self.rnn_net.weight_out
        self.hidden_weight = self.rnn_net.weight_hh
        self.out_bias = self.rnn_net.bias_out
        self.act_fcn = self.rnn_net.act_fcn

        # Set up regularization parameters
        # GPU versions for computation
        self.l1_weight = torch.tensor(hp['l1_weight'], device=self.device)
        self.l2_weight = torch.tensor(hp['l2_weight'], device=self.device)
        self.l2_firing_rate = torch.tensor(hp['l2_firing_rate'], device=self.device)
        self.l1_firing_rate = torch.tensor(hp['l1_firing_rate'], device=self.device)

        # CPU versions for condition checking (more efficient)
        self.l1_weight_cpu = torch.tensor(hp['l1_weight'], device=torch.device("cpu"))
        self.l2_weight_cpu = torch.tensor(hp['l2_weight'], device=torch.device("cpu"))
        self.l2_firing_rate_cpu = torch.tensor(hp['l2_firing_rate'], device=torch.device("cpu"))
        self.l1_firing_rate_cpu = torch.tensor(hp['l1_firing_rate'], device=torch.device("cpu"))

        # Initialize optimizer (simplified - no complex parameter management)
        self.update_optimizer()

        if self.use_piezo:
            print(f"✅ Initialized simplified training stepper with piezo")
        else:
            print("🚫 Initialized training stepper without piezo")

    def update_optimizer(self):
        """Update optimizer with all trainable parameters (simplified)"""
        # Get all trainable parameters (much simpler than before)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            print("⚠️ Warning: No trainable parameters found!")

        # Initialize the optimizer based on hyperparameters
        if self.hp['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.hp['learning_rate'])
        elif self.hp['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.hp['learning_rate'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.hp['optimizer']}. Use 'adam' or 'sgd'.")

        if self.use_piezo:
            print(f"🔧 Optimizer initialized with {len(trainable_params)} trainable parameters")

    def forward(self, inputs, initial_state, **kwargs):
        """
        Forward pass through the RNN model.

        Args:
            inputs: Input tensor of shape (time, batch_size, input_size)
            initial_state: Initial hidden state of shape (batch_size, hidden_size)
            **kwargs: Additional arguments (including hb_sequence for piezo)

        Returns:
            List of hidden states for all time steps
        """
        if self.use_piezo:
            # Enhanced forward pass with simplified piezo
            hb_sequence = kwargs.get('hb_sequence', None)
            return self.rnn_net(inputs, initial_state, hb_sequence=hb_sequence)
        else:
            # Original forward pass
            return self.rnn_net(inputs, initial_state)

    def cost_fcn(self, **kwargs):
        """
        Calculate the cost/loss function for training.

        SIMPLIFIED: No reconstruction modes, just standard cognitive task loss

        Args:
            inputs: Input tensor of shape (time, batch_size, input_size)
            target_outputs: Target tensor of shape (time, batch_size, output_size)
            cost_mask: Mask tensor indicating which timesteps to include in loss
            cost_start_time: First timestep to include in loss calculation
            cost_end_time: Last timestep to include in loss calculation
            initial_state: Initial hidden state
            seq_mask: Mask for valid sequence elements
            hb_sequence: Heartbeat sequence (for simplified piezo)

        Sets:
            self.cost: Total cost (loss + regularization)
            self.cost_lsq: Mean squared error component of loss
            self.cost_reg: Regularization component of loss
        """
        # Extract inputs from kwargs
        inputs = kwargs['inputs']
        target_outputs = kwargs['target_outputs']
        cost_mask = kwargs.get('cost_mask', torch.ones_like(target_outputs))
        cost_start_time = kwargs.get('cost_start_time', 0)
        cost_end_time = kwargs.get('cost_end_time', inputs.shape[0])
        initial_state = kwargs['initial_state']

        # Handle sequence mask
        if 'seq_mask' in kwargs:
            seq_mask = kwargs['seq_mask'].type(torch.float32).unsqueeze(2)
        else:
            seq_mask = torch.ones(inputs.shape[0], inputs.shape[1], 1, device=self.device)

        # Store dimensions for reuse
        self.batch_size, self.hidden_size = initial_state.shape

        # === FORWARD PASS AND STATE COLLECTION ===
        # Forward pass through the model to collect all states
        if self.use_piezo:
            hb_sequence = kwargs.get('hb_sequence', None)
            self.state_collector = self.forward(inputs, initial_state, hb_sequence=hb_sequence)
        else:
            self.state_collector = self.forward(inputs, initial_state)

        # Extract and reshape relevant hidden states for the time window of interest
        # We add 1 to indices because state_collector includes the initial state
        self.state_binder = torch.cat(self.state_collector[cost_start_time + 1:cost_end_time + 1], dim=0).view(
            -1, self.batch_size, self.hidden_size)
        norms = torch.norm(self.state_binder, dim=2)  # shape: [T, B]
        mean_norm = norms.mean().item()
        max_norm = norms.max().item()
        #print(f"[STATE DIAGNOSTIC] Mean Hidden State Norm: {mean_norm:.4f} | Max: {max_norm:.4f}")

        # Apply activation function to get firing rates
        self.firing_rate_binder = self.act_fcn(self.state_binder)

        # Calculate model outputs
        self.outputs = torch.matmul(self.firing_rate_binder, self.out_weight) + self.out_bias

        # === COST FUNCTION ===
        rule_name = self.hp.get("rule_name", None)

        if rule_name == "time_bisection":
            # Shapes: logits/targets: [T, B, O], cost_mask: [T, B] or [T, B, O]
            logits = self.outputs
            targets = target_outputs

            # Derive a [T, B] mask (collapse output dim if present)
            if cost_mask.dim() == 3:
                mask_tb = (cost_mask != 0).any(dim=2)  # [T, B]
            else:
                mask_tb = cost_mask  # [T, B]

            # Response step = first time the targets become non-zero
            targets_active = (targets.abs().sum(dim=2) > 0)  # [T, B]
            has_target = targets_active.any(dim=0)  # [B]
            response_idx_targets = targets_active.float().argmax(dim=0)  # [B]
            # Fallback: if a trial has all-zero targets, use the first masked step
            response_idx_mask = mask_tb.float().argmax(dim=0)  # [B]
            response_idx = torch.where(has_target, response_idx_targets, response_idx_mask)  # [B]

            # Batch-first for gather
            logits_b = logits.permute(1, 0, 2)  # [B, T, O]
            targets_b = targets.permute(1, 0, 2)  # [B, T, O]

            # Gather per-trial step
            response_idx_exp = response_idx.view(-1, 1, 1).expand(-1, 1, logits_b.size(2))  # [B, 1, O]
            chosen_logits = logits_b.gather(1, response_idx_exp).squeeze(1)  # [B, O]
            chosen_targets = targets_b.gather(1, response_idx_exp).squeeze(1)  # [B, O]

            # One-hot → class indices
            target_classes = chosen_targets.argmax(dim=1)  # [B]

            # CE loss at the chosen step
            self.cost_lsq = torch.nn.functional.cross_entropy(chosen_logits, target_classes)
        else:
            # Default: mean squared error with mask (all other tasks)
            cost_mask_length = torch.sum(cost_mask, dim=0)
            cost_mask_length = torch.clamp(cost_mask_length, min=1.0)  # Avoid division by zero
            self.cost_lsq = torch.mean(
                torch.sum(((self.outputs - target_outputs) ** self._2) * cost_mask, dim=0) / cost_mask_length)

        # Calculate regularization cost
        self.cost_reg = self._0

        # L1 weight regularization
        if self.l1_weight_cpu > 0:
            temp = self._0
            for x in self.weight_list:
                temp = temp + torch.mean(torch.abs(x))
            self.cost_reg = self.cost_reg + temp * self.l1_weight

        # L2 weight regularization
        if self.l2_weight_cpu > 0:
            temp = self._0
            for x in self.weight_list:
                temp = temp + torch.mean(x ** self._2)
            self.cost_reg = self.cost_reg + temp * self.l2_weight

        # Firing rate regularization
        if self.firing_rate_binder is not None:
            # L2 firing rate regularization
            if self.l2_firing_rate_cpu > 0:
                seq_mask_n_element = torch.sum(seq_mask, dim=0)
                seq_mask_n_element = torch.clamp(seq_mask_n_element, min=1.0)
                self.cost_reg = self.cost_reg + torch.mean(
                    torch.sum((self.firing_rate_binder * seq_mask) ** self._2, dim=0) / seq_mask_n_element
                ) * self.l2_firing_rate

            # L1 firing rate regularization
            if self.l1_firing_rate_cpu > 0:
                seq_mask_n_element = torch.sum(seq_mask, dim=0)
                seq_mask_n_element = torch.clamp(seq_mask_n_element, min=1.0)
                self.cost_reg = self.cost_reg + torch.mean(
                    torch.sum(torch.abs(self.firing_rate_binder * seq_mask), dim=0) / seq_mask_n_element
                ) * self.l1_firing_rate

        # Total cost is the sum of MSE loss and regularization
        self.cost = self.cost_lsq + self.cost_reg

    def stepper(self, **kwargs):
        """
        Perform one training step.

        This method:
        1. Zeros gradients
        2. Computes the cost/loss
        3. Performs backpropagation
        4. Applies gradient clipping
        5. Updates parameters
        6. Applies weight constraints

        Args:
            Same arguments as cost_fcn

        Returns:
            None (model parameters are updated in-place)
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Compute cost function and backpropagate
        self.cost_fcn(**kwargs)
        self.cost.backward()

        # Apply gradient clipping based on loss value
        if self.cost > 0.1:
            torch.nn.utils.clip_grad_value_(self.rnn_net.parameters(), self._1)
        else:
            torch.nn.utils.clip_grad_norm_(self.rnn_net.parameters(), self._1, 2)
        # Always use aggressive clipping for larger networks
        #torch.nn.utils.clip_grad_norm_(self.rnn_net.parameters(), 0.5, 2)

        # Update parameters
        self.optimizer.step()

        # Apply weight constraints
        self.rnn_net.self_weight_clipper()
        self.step_count += 1

        # Print piezo connectivity diagnostic every 200 steps
        if self.use_piezo and self.step_count % 200 == 0:
            if hasattr(self.model, 'piezo_connectivity') and self.model.piezo_connectivity is not None:
                # Use the same activation as in network.py (tanh + relu)
                conn_strength = torch.tanh(self.model.piezo_connectivity)
                conn_strength = torch.relu(conn_strength)

                active_connections = (conn_strength > 0.01).sum().item()
                total_connections = len(conn_strength)

                # Check if parameter is being updated
                has_grad = self.model.piezo_connectivity.grad is not None
                grad_norm = self.model.piezo_connectivity.grad.norm().item() if has_grad else 0.0
                requires_grad = self.model.piezo_connectivity.requires_grad

                print(f"[PIEZO DIAGNOSTIC] Step {self.step_count}: "
                      f"{active_connections}/{total_connections} active connections, "
                      f"Mean: {conn_strength.mean():.4f}, "
                      f"Max: {conn_strength.max():.4f}, "
                      f"Min: {conn_strength.min():.4f}")
                print(f"                   Requires_grad: {requires_grad}, "
                      f"Has_grad: {has_grad}, Grad_norm: {grad_norm:.6f}")

    def get_loss_components(self):
        """
        Get the components of the loss function.

        Returns:
            dict: Dictionary containing loss components
        """
        return {
            'total': self.cost.item(),
            'mse': self.cost_lsq.item(),
            'reg': self.cost_reg.item()
        }

    def get_info(self):
        """Get information about current training setup"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        info = {
            'mode': self.mode,
            'use_piezo': self.use_piezo,
            'num_trainable_params': len(trainable_params),
            'simplified': True
        }

        if self.use_piezo:
            # Check if connectivity is trainable
            if hasattr(self.model, 'piezo_connectivity') and self.model.piezo_connectivity is not None:
                info['connectivity_trainable'] = self.model.piezo_connectivity.requires_grad
            else:
                info['connectivity_trainable'] = False

        return info