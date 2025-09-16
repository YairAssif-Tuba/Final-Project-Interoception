"""
Insula Module for Heartbeat Detection

This module provides an insula RNN that processes ECG signals and outputs aINS activity.
It expects ECG data in the format from ecg_libraries_hr_cal_split/ (numpy arrays).

Usage:
    from insula_module import InsulaModule
    
    # Initialize module
    insula = InsulaModule(device='cuda')
    
    # Process single ECG signal (from ecg_libraries format)
    ecg_signal = np.array([...])  # ECG samples at 100Hz, shape [T]
    aINS_activity = insula.process_ecg(ecg_signal)  # Returns [T, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math
import numpy as np
from typing import Optional, Tuple, List


class InsulaModule(nn.Module):
    """
    Insula RNN module for heartbeat processing.
    
    This module contains the insula subnetwork (32 units: 11 gINS + 11 dINS + 10 aINS)
    with frozen pretrained weights. It processes ECG signals and outputs aINS activity.
    """
    
    def __init__(self, device: str = 'cuda', weights_path: Optional[str] = None, config_path: Optional[str] = None, freeze: bool = True, load_pretrained: bool = True):
        """
        Initialize the insula module.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            weights_path: Path to weights file (default: './insula_weights.pt')
            config_path: Path to config file (default: './config.json')
        """
        super().__init__()
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract architecture parameters
        arch = self.config['model_architecture']
        dynamics = self.config['dynamics']
        
        self.n_input = arch['n_input']
        self.n_insula = arch['n_insula']
        self.n_gINS = arch['n_gINS']
        self.n_dINS = arch['n_dINS']
        self.n_aINS = arch['n_aINS']
        self.activation_name = arch['activation']
        
        # Dynamics parameters
        self.dt_ms = dynamics['dt_ms']
        self.tau_ms = dynamics['tau_ms']
        self.alpha = torch.tensor(dynamics['alpha'], device=self.device)
        self.sigma_rec = torch.tensor(dynamics['sigma_rec'], device=self.device)
        
        # ECG processing parameters
        self.fs = self.config['ecg_processing']['fs']
        self.zscore_eps = self.config['ecg_processing']['zscore_eps']
        
        # Peak detection parameters
        self.peak_thr = self.config['peak_detection']['peak_thr']
        self.peak_min_dist_ms = self.config['peak_detection']['peak_min_dist_ms']
        
        # Define insula indices
        self.gINS_idx = list(range(0, self.n_gINS))
        self.dINS_idx = list(range(self.n_gINS, self.n_gINS + self.n_dINS))
        self.aINS_idx = list(range(self.n_gINS + self.n_dINS, self.n_gINS + self.n_dINS + self.n_aINS))
        self.insula_idx = self.gINS_idx + self.dINS_idx + self.aINS_idx
        
        # Set activation function
        if self.activation_name == 'tanh':
            self.act_fcn = torch.tanh
        elif self.activation_name == 'relu':
            self.act_fcn = F.relu
        elif self.activation_name == 'softplus':
            self.act_fcn = F.softplus
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        
        # Initialize network parameters (will be loaded from weights unless training from scratch)
        # Note: weight_ih is not used since we only have heartbeat input
        self.weight_hh = nn.Parameter(torch.empty(self.n_insula, self.n_insula))
        self.bias_h = nn.Parameter(torch.empty(1, self.n_insula))
        self.reset_parameters()
        
        # Create structural mask for gINS → dINS → aINS connectivity
        self.structural_mask = None
        self._create_structural_mask()
        
        # Load pretrained weights (optional)
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), 'insula_weights.pt')
        if load_pretrained:
            if os.path.exists(weights_path):
                self.load_weights(weights_path)
            else:
                raise FileNotFoundError(f"Requested to load_pretrained=True but weights not found at {weights_path}")
        
        # Move to device
        self.to(self.device)
        
        # Apply structural mask after moving to device
        self.apply_structural_mask()
        
        # Freeze or unfreeze parameters for training
        self.set_freeze(freeze)
    
    def load_weights(self, weights_path: str):
        """Load pretrained weights from file."""
        weights = torch.load(weights_path, map_location=self.device)
        
        # Load weights (weight_ih is not used since we only have heartbeat input)
        self.weight_hh.data = weights['weight_hh']
        self.bias_h.data = weights['bias_h']
        
        print(f"Loaded insula weights from {weights_path}")
        print(f"  weight_hh: {self.weight_hh.shape}")
        print(f"  bias_h: {self.bias_h.shape}")
    
    def reset_parameters(self):
        """Initialize recurrent weights like original RNN (diag ~0.999, off-diag Gaussian)."""
        N = int(self.n_insula)
        # Use same effective std as original 256-neuron network: 0.3/sqrt(256) = 0.01875
        std = 0.01875
        hh_mask = torch.ones(N, N) - torch.eye(N)
        non_diag = torch.empty(N, N).normal_(0.0, std)
        weight_hh = torch.eye(N) * 0.999 + hh_mask * non_diag
        with torch.no_grad():
            self.weight_hh.copy_(weight_hh)
            self.bias_h.zero_()
    
    def _create_structural_mask(self):
        """Create structural mask for insula connectivity based on config."""
        # Get layer sizes from config
        n_gINS = self.n_gINS
        n_dINS = self.n_dINS  
        n_aINS = self.n_aINS
        total = self.n_insula
        
        # Verify consistency
        if total != (n_gINS + n_dINS + n_aINS):
            raise ValueError(f"Config inconsistent: n_insula={total} != n_gINS+n_dINS+n_aINS={n_gINS+n_dINS+n_aINS}")
        
        # Initialize all-zeros mask (no connections)
        mask = torch.zeros(total, total, dtype=torch.float32)
        
        # Define layer indices
        gINS_start, gINS_end = 0, n_gINS
        dINS_start, dINS_end = n_gINS, n_gINS + n_dINS
        aINS_start, aINS_end = n_gINS + n_dINS, total
        
        # Fixed connectivity rules: gINS → gINS+dINS, dINS → dINS+aINS, aINS → aINS
        # gINS connections: gINS → gINS, gINS → dINS
        mask[gINS_start:gINS_end, gINS_start:gINS_end] = 1.0  # gINS → gINS
        mask[gINS_start:gINS_end, dINS_start:dINS_end] = 1.0  # gINS → dINS
        
        # dINS connections: dINS → dINS, dINS → aINS
        mask[dINS_start:dINS_end, dINS_start:dINS_end] = 1.0  # dINS → dINS
        mask[dINS_start:dINS_end, aINS_start:aINS_end] = 1.0  # dINS → aINS
        
        # aINS connections: aINS → aINS (final layer, self-connections only)
        mask[aINS_start:aINS_end, aINS_start:aINS_end] = 1.0  # aINS → aINS
        
        self.structural_mask = mask
        
        # Store indices for reference
        self.gINS_indices = list(range(gINS_start, gINS_end))
        self.dINS_indices = list(range(dINS_start, dINS_end))
        self.aINS_indices = list(range(aINS_start, aINS_end))
        
        print(f"Created structural mask from config: gINS={n_gINS}, dINS={n_dINS}, aINS={n_aINS}")
        print(f"Fixed connectivity: gINS→gINS+dINS, dINS→dINS+aINS, aINS→aINS")
    
    def apply_structural_mask(self):
        """Apply structural mask to enforce connectivity constraints."""
        if self.structural_mask is not None:
            with torch.no_grad():
                # Move mask to same device as weights
                mask = self.structural_mask.to(self.weight_hh.device)
                self.weight_hh.data *= mask
    
    
    def save_weights(self, weights_path: str):
        """Save current insula weights to file (compatible with load_weights)."""
        payload = {
            'weight_hh': self.weight_hh.detach().to('cpu'),
            'bias_h': self.bias_h.detach().to('cpu'),
        }
        torch.save(payload, weights_path)
        print(f"Saved insula weights to {weights_path}")
    
    def set_freeze(self, freeze: bool):
        """Freeze or unfreeze all parameters and set mode accordingly."""
        for p in self.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.eval()
        else:
            self.train()
    
    def zscore_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalize ECG signal.
        
        Args:
            ecg: ECG signal [T]
            
        Returns:
            Z-scored ECG signal [T]
        """
        mean = ecg.mean()
        std = ecg.std()
        return (ecg - mean) / (std + self.zscore_eps)
    
    def forward(self, initial_state: torch.Tensor, heartbeat: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the insula network.
        
        Args:
            initial_state: Initial hidden state [1, n_insula]
            heartbeat: ECG signal [T, 1]
            
        Returns:
            List of hidden states for all time steps
        """
        state = initial_state
        state_collector = [state]
        
        # Process each time step
        for t, hb_t in enumerate(heartbeat):
            # Add heartbeat to gINS units
            state = state.clone()
            state[:, self.gINS_idx] = state[:, self.gINS_idx] + hb_t.unsqueeze(1)
            
            # Recurrent update
            state_new = (torch.matmul(state, self.weight_hh) + self.bias_h + 
                        torch.randn_like(state, device=self.device) * self.sigma_rec)
            
            # Leaky integration
            state = (1 - self.alpha) * state + self.alpha * state_new
            
            # Apply activation
            state = self.act_fcn(state)
            
            state_collector.append(state)
        
        return state_collector
    
    def forward_batch(self, heartbeat_tb: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Batched forward pass through the insula network.
        
        Args:
            heartbeat_tb: Heartbeat input [T, B] (time first)
            initial_state: Optional initial hidden state [B, n_insula]
        
        Returns:
            States over time [T, B, n_insula]
        """
        if heartbeat_tb.dim() != 2:
            raise ValueError(f"Expected heartbeat_tb shape [T,B], got {tuple(heartbeat_tb.shape)}")
        T, B = int(heartbeat_tb.shape[0]), int(heartbeat_tb.shape[1])
        if initial_state is None:
            state = torch.zeros(B, self.n_insula, device=self.device)
        else:
            state = initial_state.to(self.device)
            if state.shape != (B, self.n_insula):
                raise ValueError(f"initial_state must be [B,{self.n_insula}], got {tuple(state.shape)}")
        states_tbn: List[torch.Tensor] = []
        for t in range(T):
            hb_t = heartbeat_tb[t].unsqueeze(1).to(self.device)  # [B,1]
            state = state.clone()
            state[:, self.gINS_idx] = state[:, self.gINS_idx] + hb_t
            state_new = (torch.matmul(state, self.weight_hh) + self.bias_h +
                         torch.randn_like(state, device=self.device) * self.sigma_rec)
            state = (1 - self.alpha) * state + self.alpha * state_new
            state = self.act_fcn(state)
            states_tbn.append(state)
        return torch.stack(states_tbn, dim=0)  # [T,B,n_insula]
    
    def process_ecg(self, ecg: np.ndarray) -> torch.Tensor:
        """
        Process ECG signal and return aINS activity.
        
        Args:
            ecg: ECG signal from ecg_libraries format: np.array([T])
            
        Returns:
            aINS activity [T, n_aINS]
        """
        with torch.no_grad():
            # Convert numpy to torch tensor and move to device
            if isinstance(ecg, np.ndarray):
                ecg_tensor = torch.from_numpy(ecg).float().to(self.device)
            else:
                ecg_tensor = ecg.float().to(self.device)
            
            # Ensure single signal format
            if ecg_tensor.dim() != 1:
                raise ValueError(f"Expected 1D ECG signal [T], got shape {ecg_tensor.shape}")
            
            T = ecg_tensor.shape[0]  # T=time_steps
            
            # Z-score normalize
            ecg_norm = self.zscore_ecg(ecg_tensor)
            
            # Prepare for forward pass
            initial_state = torch.zeros(1, self.n_insula, device=self.device)  # [1, 32]
            heartbeat = ecg_norm.unsqueeze(0).t()  # [T, 1] - transpose for time-first format
            
            # Forward pass
            states = self.forward(initial_state, heartbeat)
            
            # Extract aINS activity (last 10 units)
            aINS_states = torch.stack([state[0, self.aINS_idx] for state in states[1:]], dim=0)  # [T, 10]
            
            return aINS_states
    
    def process_ecg_batch(self, batch_ecg_bt: np.ndarray) -> torch.Tensor:
        """
        Process a batch of ECG signals and return aINS activity.
        
        Args:
            batch_ecg_bt: ECG batch [B, T] as np.ndarray or torch.Tensor
        
        Returns:
            aINS activity [T, B, n_aINS]
        """
        if isinstance(batch_ecg_bt, np.ndarray):
            ecg_bt = torch.from_numpy(batch_ecg_bt).float().to(self.device)
        else:
            ecg_bt = batch_ecg_bt.float().to(self.device)
        if ecg_bt.dim() != 2:
            raise ValueError(f"Expected batch ECG [B,T], got shape {tuple(ecg_bt.shape)}")
        # Per-window z-score along time
        eps = 1e-6
        mean_bt = ecg_bt.mean(dim=1, keepdim=True)
        std_bt = ecg_bt.std(dim=1, keepdim=True)
        ecg_norm_bt = (ecg_bt - mean_bt) / (std_bt + eps)
        hb_tb = ecg_norm_bt.t()  # [T,B]
        states_tbn = self.forward_batch(hb_tb)  # [T,B,N]
        aINS_tba = states_tbn[:, :, self.aINS_idx]  # [T,B,A]
        return aINS_tba
    


def create_beat_head(weights_path: Optional[str] = None) -> nn.Module:
    """
    Create a beat head module for converting aINS activity to beat probabilities.
    
    Args:
        weights_path: Path to beat head weights (default: './beat_head_weights.pt')
        
    Returns:
        Beat head module
    """
    beat_head = nn.Linear(10, 1)  # 10 aINS units -> 1 output
    
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), 'beat_head_weights.pt')
    
    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        # Map proj.weight/proj.bias to weight/bias for nn.Linear
        state_dict = {
            'weight': weights['proj.weight'],
            'bias': weights['proj.bias']
        }
        beat_head.load_state_dict(state_dict)
        print(f"Loaded beat head weights from {weights_path}")
    
    return beat_head

