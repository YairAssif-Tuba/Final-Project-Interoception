"""
Neural Network for Temporal Processing Tasks - AMENDED VERSION

Key changes:
- Time delay in piezo interface
- No adaptation mechanism
- Clean integration for 2M trial training
"""

import torch
from torch import nn
import os
import math
import numpy as np

# Import amended piezo interface
try:
    from simple_piezo import SimplePiezoInterface
    PIEZO_AVAILABLE = True
except ImportError:
    PIEZO_AVAILABLE = False


class RNN(nn.Module):
    """RNN with amended piezo interface"""

    def __init__(self, hp, is_cuda=True, **kwargs):
        super(RNN, self).__init__()

        input_size = hp['n_input']
        hidden_size = hp['n_rnn']
        output_size = hp['n_output']
        alpha = hp['alpha']
        sigma_rec = hp['sigma_rec']
        act_fcn = hp['activation']

        self.hp = hp
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if is_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Activation function
        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)
        elif act_fcn == 'tanh':
            self.act_fcn = lambda x: torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {act_fcn}")

        # AMENDED PIEZO SETUP
        self.use_piezo = hp.get('use_piezo', False)

        if self.use_piezo:
            if not PIEZO_AVAILABLE:
                raise ImportError("SimplePiezoInterface not available")

            self.piezo = SimplePiezoInterface(
                num_neurons=hidden_size,
                connection_fraction=hp.get('piezo_connection_fraction', 0.15),
                slice_size=hp.get('heartbeat_slice_size', 20),
                use_temporal_delay=hp.get('use_temporal_delay', False),  # Enable by default
                delay_steps=hp.get('temporal_delay_steps', 3)
            ).to(self.device)

            # Trainable connectivity parameter
            self.piezo_connectivity = nn.Parameter(
                torch.randn(self.piezo.num_connected, device=self.device) * 0.3 + 1.0
            )
            self.piezo_connected_indices = self.piezo.connected_indices

            print(f"✅ Amended piezo interface: {self.piezo.num_connected} receptors, time_delay={hp.get('use_temporal_delay', True)}")
        else:
            self.piezo = None
            self.piezo_connectivity = None
            print("🚫 Piezo interface disabled")

        # Task-specific input weights
        rule_name = kwargs.get('rule_name', None)

        if rule_name == 'interval_production':
            if input_size != 2:
                raise ValueError('input_size should be 2 for interval_production')
            self.weight_ih = nn.Parameter(
                torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(1), 1./math.sqrt(1))
            )
        elif rule_name in ['interval_comparison', 'time_bisection']:
            if input_size != 2:
                raise ValueError(f'input_size should be 2 for {rule_name}')
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            self.weight_ih = nn.Parameter(weight_ih)
        elif rule_name == 'gaussian_bisection':
            if input_size != 1:
                raise ValueError('input_size should be 1 for gaussian_bisection')
            # Use the same initialization as interval_comparison since both involve duration discrimination
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            self.weight_ih = nn.Parameter(weight_ih)
        else:
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            self.weight_ih = nn.Parameter(weight_ih)

        # Recurrent weights (Bi & Zhou's approach)
        hh_mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
        non_diag = torch.empty(hidden_size, hidden_size).normal_(
            0, hp['initial_std']/math.sqrt(hidden_size)
        )
        weight_hh = torch.eye(hidden_size)*0.999 + hh_mask * non_diag

        self.weight_hh = nn.Parameter(weight_hh)
        self.bias_h = nn.Parameter(torch.zeros(1, hidden_size))
        self.weight_out = nn.Parameter(
            torch.empty(hidden_size, output_size).normal_(0., 1.0/math.sqrt(hidden_size))
        )
        self.bias_out = nn.Parameter(torch.zeros(output_size,))

        # Integration and noise
        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2./alpha) * sigma_rec, device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

    def extract_pressure_slices(self, hb_sequence, T):
        """Extract pressure slices from heartbeat sequence"""
        if hb_sequence is None:
            return None, None

        pressure = hb_sequence[:, 1]  # Column 1 is pressure
        min_pressure = torch.min(pressure)

        slice_size = self.piezo.slice_size
        pressure_slices = []

        for step in range(T):
            start = step * slice_size
            end = (step + 1) * slice_size
            if end <= len(pressure):
                slice_data = pressure[start:end]
            else:
                slice_data = pressure[start:]
                if len(slice_data) < slice_size:
                    padding = torch.zeros(slice_size - len(slice_data), device=pressure.device)
                    slice_data = torch.cat([slice_data, padding])

            pressure_slices.append(slice_data)

        return pressure_slices, min_pressure

    def forward(self, inputs, initial_state, hb_sequence=None, mode=None):
        """Forward pass with amended piezo interface"""
        T = inputs.shape[0]

        # Original logic when piezo disabled
        if not self.use_piezo:
            state = initial_state
            state_collector = [state]

            for input_per_step in inputs:
                state_new = torch.matmul(self.act_fcn(state), self.weight_hh) + self.bias_h + \
                            torch.matmul(input_per_step, self.weight_ih) + \
                            torch.randn_like(state, device=self.device) * self.sigma_rec

                state = (self._1 - self.alpha) * state + self.alpha * state_new
                state_collector.append(state)

            return state_collector

        # AMENDED PIEZO LOGIC WITH TIME DELAY
        else:
            pressure_slices, min_pressure = self.extract_pressure_slices(hb_sequence, T)

            state = initial_state
            state_collector = [state]

            for t in range(T):
                input_per_step = inputs[t]

                h_act = self.act_fcn(state)
                recurrent_term = torch.matmul(h_act, self.weight_hh)
                bias_term = self.bias_h
                input_term = torch.matmul(input_per_step, self.weight_ih)
                noise_term = torch.randn_like(state, device=self.device) * self.sigma_rec
                base_state_update = recurrent_term + bias_term + input_term + noise_term

                # Piezo modulation with time delay
                if pressure_slices is not None and min_pressure is not None:
                    pressure_slice = pressure_slices[t]
                    cardiac_responses = self.piezo(pressure_slice, min_pressure)

                    full_connectivity = torch.zeros(self.hidden_size, device=self.device)
                    connected_connectivity = torch.tanh(self.piezo_connectivity)
                    connected_connectivity = torch.relu(connected_connectivity)
                    full_connectivity[self.piezo_connected_indices] = connected_connectivity

                    piezo_mod = full_connectivity * cardiac_responses * 1.08
                else:
                    piezo_mod = 0.0

                state_new = base_state_update + piezo_mod
                state = (self._1 - self.alpha) * state + self.alpha * state_new
                state_collector.append(state)

            return state_collector

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element = self.weight_hh.diag().data.clamp_(0., 1.)
        self.weight_hh.data[range(self.hidden_size), range(self.hidden_size)] = diag_element

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.state_dict(), save_path)

    def load(self, model_dir):
        if model_dir is None:
            return False

        save_path = os.path.join(model_dir, 'model.pth')
        if os.path.isfile(save_path):
            self.load_state_dict(
                torch.load(save_path, map_location=lambda storage, loc: storage),
                strict=False
            )
            return True
        return False

    def get_piezo_info(self):
        if not self.use_piezo:
            return {'enabled': False}

        stats = self.piezo.get_statistics()
        stats['enabled'] = True
        stats['connectivity_param_trainable'] = self.piezo_connectivity.requires_grad
        return stats