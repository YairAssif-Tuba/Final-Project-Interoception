import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from simple_piezo import SimplePiezoInterface as PiezoInterface

def generate_dummy_input_vector(phi, num_harmonics=5):
    # Harmonic basis: [cos(2πkφ), sin(2πkφ)] for k = 1..num_harmonics
    harmonic_part = []
    for k in range(1, num_harmonics + 1):
        angle = 2 * math.pi * k * torch.tensor(phi)
        harmonic_part.append(torch.cos(angle))
        harmonic_part.append(torch.sin(angle))

    # Signal features: current_val, mean_val, peak_val, delta, curvature
    # First three should be positive
    current_val = torch.rand(1) * 2.0 + 0.5   # range [0.5, 2.5]
    mean_val = torch.rand(1) * 2.0 + 0.5
    peak_val = torch.rand(1) * 2.0 + 0.5
    delta = torch.randn(1) * 0.5              # can be negative
    curvature = torch.randn(1) * 0.5

    signal_part = [current_val, mean_val, peak_val, delta, curvature]
    signal_part = [v.squeeze() for v in signal_part]

    return torch.tensor(harmonic_part + signal_part)

def generate_phi_sweep(n_bins=100):
    return torch.linspace(0, 1, steps=n_bins)

def plot_forward_tuning(model, phi_values, save_prefix=""):
    outputs = []
    for phi in phi_values:
        input_vector = generate_dummy_input_vector(phi.item()).to(next(model.parameters()).device)
        out = model.forward(phi.item(), input_vector).detach().cpu().numpy()
        outputs.append(out)
    outputs = np.stack(outputs)

    plt.figure(figsize=(12, 6))
    for i in range(model.num_neurons):
        plt.plot(phi_values.numpy(), outputs[:, i], alpha=0.6)
    plt.title("Full Forward Tuning Curves")
    plt.xlabel("Phase φ")
    plt.ylabel("Output")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}forward_tuning_curves.png", dpi=300)
    plt.show()

def plot_harmonic_only_tuning(model, phi_values, save_prefix=""):
    w_harm = torch.softmax(model.phase_logits_harmonics, dim=1).detach().cpu().numpy()
    num_neurons, num_features = w_harm.shape
    num_harmonics = num_features // 2

    harmonic_curves = []
    for phi in phi_values.numpy():
        harm_input = []
        for k in range(1, num_harmonics + 1):
            harm_input.append(np.cos(2 * np.pi * k * phi))
            harm_input.append(np.sin(2 * np.pi * k * phi))
        harm_input = np.array(harm_input)
        harmonic_curves.append(np.dot(w_harm, harm_input))  # shape: [num_neurons]
    harmonic_curves = np.stack(harmonic_curves)

    plt.figure(figsize=(12, 6))
    for i in range(num_neurons):
        plt.plot(phi_values.numpy(), harmonic_curves[:, i], alpha=0.6)
    plt.title("Harmonic-Only Tuning Curves")
    plt.xlabel("Phase φ")
    plt.ylabel("Weighted Harmonic Projection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}harmonic_tuning_curves.png", dpi=300)
    plt.show()

def plot_overlay_tuning_curves(model, phi_values, save_prefix=""):
    w_harm = torch.softmax(model.phase_logits_harmonics, dim=1).detach().cpu().numpy()
    num_neurons, num_features = w_harm.shape
    num_harmonics = num_features // 2

    harmonic_curves = []
    forward_outputs = []

    for phi in phi_values:
        phi_val = phi.item()

        harm_input = []
        for k in range(1, num_harmonics + 1):
            harm_input.append(np.cos(2 * np.pi * k * phi_val))
            harm_input.append(np.sin(2 * np.pi * k * phi_val))
        harm_input = np.array(harm_input)
        harmonic_curves.append(np.dot(w_harm, harm_input))

        input_vector = generate_dummy_input_vector(phi_val).to(next(model.parameters()).device)
        out = model.forward(phi_val, input_vector).detach().cpu().numpy()
        forward_outputs.append(out)

    harmonic_curves = np.stack(harmonic_curves)
    forward_outputs = np.stack(forward_outputs)

    for i in range(model.num_neurons):
        plt.figure()
        plt.plot(phi_values.numpy(), forward_outputs[:, i], label="Full Forward", linewidth=2)
        plt.plot(phi_values.numpy(), harmonic_curves[:, i], label="Harmonic Only", linestyle="--")
        plt.title(f"Neuron {i} — Forward vs Harmonic Tuning")
        plt.xlabel("Phase φ")
        plt.ylabel("Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}neuron_{i:02d}_overlay.png", dpi=300)
        plt.close()

def main():
    num_neurons = 32
    num_harmonics = 5
    signal_feature_count = 5
    dt = 10  # ms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PiezoInterface(
        num_neurons=num_neurons,
        num_harmonics=num_harmonics,
        signal_feature_count=signal_feature_count,
        dt=dt,
        connection_fraction=1.0,
        use_temporal_delay=False
    ).to(device)
    model.eval()

    phi_values = generate_phi_sweep(100)

    print(">>> Plotting full forward-based tuning curves...")
    plot_forward_tuning(model, phi_values)

    print(">>> Plotting harmonic-only tuning curves...")
    plot_harmonic_only_tuning(model, phi_values)

    print(">>> Plotting overlay tuning curves (forward vs harmonic-only)...")
    plot_overlay_tuning_curves(model, phi_values)

if __name__ == "__main__":
    main()
