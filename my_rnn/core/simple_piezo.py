# ===================================================================
# simple_piezo.py - AMENDED VERSION
# ===================================================================

import torch
import torch.nn as nn
import numpy as np


class SimplePiezoInterface(nn.Module):
    """
    Simple piezo interface with time delay and no adaptation.
    """

    def __init__(self, num_neurons, connection_fraction=0.15, slice_size=20,
                 use_temporal_delay=False, delay_steps=3):
        super().__init__()

        self.num_neurons = num_neurons
        self.slice_size = slice_size
        self.use_temporal_delay = use_temporal_delay
        self.delay_steps = delay_steps

        # Connection pattern
        num_connected = int(num_neurons * connection_fraction)
        connected_indices = torch.randperm(num_neurons)[:num_connected]

        self.register_buffer("connected_indices", connected_indices)
        self.num_connected = num_connected

        # Individual receptor characteristics
        sigmoid_steepness = torch.empty(self.num_connected).uniform_(0.8, 3.0)
        self.register_buffer("sigmoid_steepness", sigmoid_steepness)

        sensitivity_offset = torch.empty(self.num_connected).uniform_(-0.2, 0.2)
        self.register_buffer("sensitivity_offset", sensitivity_offset)

        response_strength = torch.empty(self.num_connected).uniform_(0.7, 1.3)
        self.register_buffer("response_strength", response_strength)

        # Time delay buffers
        if self.use_temporal_delay:
            # Individual fractional delays for each receptor
            delay_times = torch.empty(self.num_connected).uniform_(1.0, 4.0)
            self.register_buffer("delay_times", delay_times)

            # History buffers
            self.history_buffers = []
            for _ in range(self.num_connected):
                buffer = torch.zeros(self.delay_steps + 2)
                self.history_buffers.append(buffer)

        print(f"SimplePiezoInterface: {num_connected}/{num_neurons} receptors, delay={use_temporal_delay}")

    def to(self, device):
        super().to(device)
        if self.use_temporal_delay:
            for i in range(len(self.history_buffers)):
                self.history_buffers[i] = self.history_buffers[i].to(device)
        return self

    def forward(self, pressure_slice, min_pressure_sequence):
        """Forward pass with time delay"""
        mean_pressure = torch.mean(pressure_slice)
        pressure_diff = mean_pressure - min_pressure_sequence

        # Individual receptor responses
        individual_inputs = pressure_diff + self.sensitivity_offset
        sigmoid_activations = torch.sigmoid(self.sigmoid_steepness * individual_inputs)
        current_responses = torch.clamp(sigmoid_activations * self.response_strength, max=1.0)

        # Apply time delays
        if self.use_temporal_delay:
            delayed_responses = self._apply_delays(current_responses)
        else:
            delayed_responses = current_responses

        # Map to full population
        full_response = torch.zeros(self.num_neurons, device=pressure_slice.device)
        full_response[self.connected_indices] = delayed_responses

        return full_response

    def _apply_delays(self, current_responses):
        """Apply fractional time delays"""
        delayed_responses = torch.zeros_like(current_responses)

        for i in range(self.num_connected):
            # Update history
            self.history_buffers[i] = torch.roll(self.history_buffers[i], -1)
            self.history_buffers[i][-1] = current_responses[i]

            # Get delayed response with fractional interpolation
            delay = self.delay_times[i].item()
            floor_delay = int(delay)
            ceil_delay = floor_delay + 1
            frac = delay - floor_delay

            if floor_delay < len(self.history_buffers[i]) and ceil_delay < len(self.history_buffers[i]):
                floor_val = self.history_buffers[i][-(floor_delay + 1)]
                ceil_val = self.history_buffers[i][-(ceil_delay + 1)]
                delayed_responses[i] = (1 - frac) * floor_val + frac * ceil_val
            else:
                delayed_responses[i] = current_responses[i]

        return delayed_responses

    def get_sequence_min_pressure(self, full_pressure_sequence):
        return torch.min(full_pressure_sequence)

    def get_statistics(self):
        return {
            'num_connected': self.num_connected,
            'connection_fraction': self.num_connected / self.num_neurons,
            'trainable_params': 0,
            'temporal_delay_enabled': self.use_temporal_delay,
            'delay_range': (
            self.delay_times.min().item(), self.delay_times.max().item()) if self.use_temporal_delay else None
        }