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

# Import insula interface (standalone, pretrained and frozen)
try:
    from standalone_insula_module.insula_module import InsulaModule
    INSULA_AVAILABLE = True
except Exception:
    INSULA_AVAILABLE = False


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

        # AMENDED PIEZO/INSULA SETUP
        self.use_piezo = hp.get('use_piezo', False)
        self.use_insula = hp.get('use_insula', False)

        # Mutual exclusivity: prefer insula if both are accidentally enabled
        if self.use_insula and self.use_piezo:
            print("⚠️ Both use_insula and use_piezo set. Preferring insula and disabling piezo.")
            self.use_piezo = False

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

        # Insula setup (pretrained + frozen)
        if self.use_insula:
            if not INSULA_AVAILABLE:
                raise ImportError("InsulaModule not available. Ensure standalone_insula_module is on PYTHONPATH.")

            weights_path = hp.get('insula_weights_path', None)
            device_str = 'cuda' if (is_cuda and torch.cuda.is_available()) else 'cpu'
            self.insula = InsulaModule(device=device_str, weights_path=weights_path, freeze=True, load_pretrained=True)

            # Projection from aINS -> RNN hidden (no bias)
            self.insula_to_rnn = nn.Linear(self.insula.n_aINS, hidden_size, bias=False)
            # Xavier init with optional scale
            scale = float(hp.get('insula_projection_init_scale', 1.0))
            nn.init.xavier_uniform_(self.insula_to_rnn.weight)
            with torch.no_grad():
                self.insula_to_rnn.weight.mul_(scale)
            # Move to device
            self.insula_to_rnn = self.insula_to_rnn.to(self.device)

            # Learnable gate to prevent early swamping
            gate_init = float(hp.get('insula_gate_init', 0.2))
            self.insula_gate = nn.Parameter(torch.tensor(gate_init, device=self.device))

            # Pooling strategy
            self.insula_pooling = hp.get('insula_pooling', 'max')  # 'max' | 'mean_logits'

            print(f"🧠 Insula interface: aINS={self.insula.n_aINS} → hidden={hidden_size}, pooling={self.insula_pooling}")
        else:
            self.insula = None
            self.insula_to_rnn = None

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
        """Forward pass with amended piezo/insula interface"""
        T = inputs.shape[0]

        # Baseline logic when no HB modulation
        if not self.use_piezo and not self.use_insula:
            state = initial_state
            state_collector = [state]

            for input_per_step in inputs:
                state_new = torch.matmul(self.act_fcn(state), self.weight_hh) + self.bias_h + \
                            torch.matmul(input_per_step, self.weight_ih) + \
                            torch.randn_like(state, device=self.device) * self.sigma_rec

                state = (self._1 - self.alpha) * state + self.alpha * state_new
                state_collector.append(state)

            return state_collector

        # INSULA LOGIC (pretrained interface with dense projection)
        if self.use_insula:
            # Build per-step insula modulation sequence [T, hidden_size]
            insula_mod_seq = None

            if hb_sequence is not None:
                # Extract ECG column (0)
                ecg = hb_sequence[:, 0]
                if hasattr(ecg, 'device'):
                    ecg = ecg.to(self.device)

                # Determine current cardiac sampling rate from slice size and dt
                slice_size = int(self.hp.get('heartbeat_slice_size', 20))
                current_fs = slice_size / (self.hp['dt'] / 1000.0)  # Hz
                target_fs = float(self.hp.get('insula_target_fs', self.insula.fs) or self.insula.fs)

                # Decimate to target_fs using FIR low-pass (Hamming windowed-sinc) + stride
                factor = max(1, int(round(current_fs / target_fs)))
                total = (ecg.shape[0] // factor) * factor
                ecg_use = ecg[:total]

                if factor > 1 and total >= int(self.hp.get('insula_decimate_taps', 31)):
                    num_taps = int(self.hp.get('insula_decimate_taps', 31))
                    cutoff_hz = float(self.hp.get('insula_decimate_cutoff_hz', 40.0))
                    cutoff_hz = min(cutoff_hz, 0.45 * target_fs)  # safety
                    # Design symmetric linear-phase FIR
                    dtype = ecg_use.dtype
                    n = torch.arange(num_taps, device=self.device, dtype=dtype)
                    m = n - (num_taps - 1) / 2.0
                    fc = torch.tensor(cutoff_hz / current_fs, device=self.device, dtype=dtype)
                    hd = 2 * fc * torch.sinc(2 * fc * m)
                    if str(self.hp.get('insula_decimate_window', 'hamming')).lower() == 'hamming':
                        w = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (num_taps - 1))
                    else:
                        # default to Hamming if unknown
                        w = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (num_taps - 1))
                    h = hd * w
                    h = h / (h.sum() + 1e-8)  # DC gain = 1

                    # Convolution with padding to cancel group delay
                    pad = (num_taps - 1) // 2
                    ecg_filt = torch.nn.functional.conv1d(
                        ecg_use.contiguous().view(1, 1, -1), h.view(1, 1, -1), padding=pad
                    ).view(-1)
                    # Stride decimation
                    ecg_ds = ecg_filt[::factor]
                else:
                    # Fallback: raw stride if too short
                    ecg_ds = ecg_use[::factor]

                # Run insula to get aINS activity [T_insula, n_aINS]
                with torch.no_grad():
                    aINS_ta = self.insula.process_ecg(ecg_ds)
                    # Ensure device and dtype match state
                    aINS_ta = aINS_ta.to(self.device, dtype=initial_state.dtype)

                # Pool to RNN steps (e.g., 2×10ms per 20ms)
                steps_per_rnn = max(1, int(round(self.hp['dt'] / self.insula.dt_ms)))

                # Ensure sufficient coverage and bound length
                max_needed = T * steps_per_rnn
                if aINS_ta.shape[0] < max_needed and aINS_ta.shape[0] > 0:
                    pad_len = max_needed - aINS_ta.shape[0]
                    aINS_ta = torch.cat([aINS_ta, aINS_ta[-1:].repeat(pad_len, 1)], dim=0)
                elif aINS_ta.shape[0] > max_needed:
                    aINS_ta = aINS_ta[:max_needed]
                pooled = []
                for step in range(T):
                    s = step * steps_per_rnn
                    e = min((step + 1) * steps_per_rnn, aINS_ta.shape[0])
                    if s >= e:
                        pooled.append(torch.zeros(self.insula.n_aINS, device=self.device))
                        continue
                    window = aINS_ta[s:e]
                    if self.insula_pooling == 'max':
                        pooled.append(window.max(dim=0).values)
                    else:  # 'mean_logits' or fallback
                        pooled.append(window.mean(dim=0))

                aINS_per_step = torch.stack(pooled, dim=0)  # [T, n_aINS]
                # Optional zero-mean centering
                if bool(self.hp.get('insula_centering', False)) and aINS_per_step.numel() > 0:
                    aINS_per_step = aINS_per_step - aINS_per_step.mean(dim=0, keepdim=True)

                insula_mod_seq = self.insula_to_rnn(aINS_per_step) * self.insula_gate  # [T, hidden]
                
                # Optional diagnostic logging (only if enabled)
                if self.hp.get('insula_debug_logging', False):
                    from .insula_diagnostics import log_insula_metrics
                    log_insula_metrics(self, self.hp, hb_sequence, insula_mod_seq)

            state = initial_state
            state_collector = [state]

            for t in range(T):
                input_per_step = inputs[t]

                h_act = self.act_fcn(state)
                recurrent_term = torch.matmul(h_act, self.weight_hh)
                bias_term = self.bias_h
                input_term = torch.matmul(input_per_step, self.weight_ih)
                noise_term = torch.randn_like(state) * self.sigma_rec
                base_state_update = recurrent_term + bias_term + input_term + noise_term

                insula_mod = insula_mod_seq[t] if insula_mod_seq is not None else 0.0

                state_new = base_state_update + insula_mod
                state = (self._1 - self.alpha) * state + self.alpha * state_new
                state_collector.append(state)

            return state_collector

        # PIEZO LOGIC WITH TIME DELAY
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

    def to(self, device):
        """Move all parameters to the specified device."""
        super().to(device)
        self.device = device
        # Move scalar tensors to device
        self.alpha = self.alpha.to(device)
        
        # Fix sigma_rec if it's not a tensor
        if not isinstance(self.sigma_rec, torch.Tensor):
            self.sigma_rec = torch.tensor(self.sigma_rec, device=device)
        else:
            self.sigma_rec = self.sigma_rec.to(device)
        return self