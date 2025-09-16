#!/usr/bin/env python3
"""
Example usage of the standalone insula module with real ECG data.

This script demonstrates how to use the insula module for heartbeat detection
using the actual ECG libraries data, similar to smoke_test.py.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insula_module import InsulaModule


def parse_args():
    p = argparse.ArgumentParser(description='Standalone insula module example with real ECG data')
    p.add_argument('--ecg_library_path', type=str, 
                   default='/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/prj_interoception_modeling/ecg_libraries_hr_cal_split/ecg_lib_test.npy',
                   help='Path to ECG library file')
    p.add_argument('--checkpoint_dir', type=str, 
                   default='/storage/pblab_shared_code/PYTHON/prj_interoception_modeling_danielle/prj_interoception_modeling/runs/standalone_insula_train',
                   help='Path to checkpoint directory with trained weights')
    p.add_argument('--sampling_rate', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--sigma_ms', type=float, default=30.0)
    p.add_argument('--tolerance_ms', type=float, default=50.0)
    p.add_argument('--zthr', type=float, default=2.5)
    p.add_argument('--refractory_ms', type=float, default=300.0)
    p.add_argument('--peak_thr', type=float, default=0.5)
    p.add_argument('--peak_min_dist_ms', type=float, default=300.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--plot_zscore', action='store_true', help='Z-score ECG for plotting only')
    p.add_argument('--save_path', type=str, default='insula_example_results.png')
    p.add_argument('--hrv_scale', type=float, default=None, help='Filter samples by HRV scale')
    return p.parse_args()


def load_ecg_library(path: str, hrv_scale: float = None):
    """Load ECG library and return samples and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ECG library not found: {path}")
    
    print(f"🔍 Loading ECG library from: {path}")
    lib = np.load(path, allow_pickle=True).item()
    all_ecg_list = list(lib.get('ecg_samples', []))
    all_meta = list(lib.get('metadata', [{} for _ in all_ecg_list]))
    
    print(f"  Total ECG samples loaded: {len(all_ecg_list)}")
    if len(all_ecg_list) > 0:
        first_ecg = all_ecg_list[0]
        print(f"  First ECG sample: shape={first_ecg.shape}, dtype={first_ecg.dtype}")
        print(f"  First ECG sample min/max/mean: {first_ecg.min():.6f} / {first_ecg.max():.6f} / {first_ecg.mean():.6f}")
    
    if not all_ecg_list:
        raise ValueError("ECG library missing 'ecg_samples'")
    
    # Filter by HRV scale if specified
    if hrv_scale is not None:
        print(f"  Filtering by hrv_scale={hrv_scale}")
        filtered_indices = []
        for i, meta in enumerate(all_meta):
            if meta.get('hrv_scale') == hrv_scale:
                filtered_indices.append(i)
        
        print(f"  Found {len(filtered_indices)} samples with hrv_scale={hrv_scale}")
        if not filtered_indices:
            raise ValueError(f"No samples found with hrv_scale={hrv_scale}")
        
        ecg_list = [all_ecg_list[i] for i in filtered_indices]
        meta = [all_meta[i] for i in filtered_indices]
    else:
        ecg_list = all_ecg_list
        meta = all_meta
    
    return ecg_list, meta


def resample_1d(signal: np.ndarray, src_fs: int, target_fs: int, out_len: int = None) -> np.ndarray:
    """Simple 1D resampling using linear interpolation."""
    if src_fs == target_fs:
        return signal[:out_len] if out_len else signal
    
    src_len = len(signal)
    if out_len is None:
        out_len = int(round(src_len * target_fs / src_fs))
    
    src_indices = np.linspace(0, src_len - 1, out_len)
    return np.interp(src_indices, np.arange(src_len), signal).astype(np.float32)


def detect_rpeaks_simple(ecg: np.ndarray, fs: int, zthr: float = 2.5, refractory_ms: int = 300) -> np.ndarray:
    """Simple R-peak detection using z-score thresholding."""
    if len(ecg) < 3:
        return np.array([], dtype=np.int64)
    z = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
    refractory = max(1, int(round(refractory_ms * fs / 1000.0)))
    peaks = []
    i = 1
    N = len(z)
    while i < N - 1:
        if z[i] > zthr and z[i] >= z[i - 1] and z[i] >= z[i + 1]:
            if not peaks or (i - peaks[-1]) >= refractory:
                peaks.append(i)
                i += refractory
                continue
        i += 1
    return np.array(peaks, dtype=np.int64)


def make_event_targets(batch_ecg: np.ndarray, fs: int, sigma_ms: float,
                      zthr: float = 2.5, refractory_ms: int = 300) -> tuple:
    """Create smoothed event targets from ECG batch."""
    B, T = batch_ecg.shape
    all_peaks = []
    base = np.zeros((B, T), dtype=np.float32)
    
    for b in range(B):
        peaks = detect_rpeaks_simple(batch_ecg[b], fs=fs, zthr=zthr, refractory_ms=refractory_ms)
        all_peaks.append(peaks)
        if peaks.size:
            peaks_clamped = peaks[(peaks >= 0) & (peaks < T)]
            base[b, peaks_clamped] = 1.0
    
    # Create Gaussian kernel for smoothing
    sigma_samples = (sigma_ms / 1000.0) * fs
    sigma = max(1e-6, float(sigma_samples))
    radius = int(max(1, round(6 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.max()
    
    # Apply convolution for smoothing
    from scipy import ndimage
    smoothed = np.zeros_like(base)
    for b in range(B):
        smoothed[b] = ndimage.convolve1d(base[b], kernel, mode='constant')
    
    smoothed = np.clip(smoothed, 0.0, 1.0)
    return smoothed, all_peaks


def load_trained_model(checkpoint_dir: str, device: str):
    """Load trained insula and beat_head from checkpoint directory."""
    insula_path = os.path.join(checkpoint_dir, 'insula_weights_best.pt')
    beat_head_path = os.path.join(checkpoint_dir, 'beat_head_weights_best.pt')
    config_path = os.path.join(checkpoint_dir, 'config_best.json')
    
    if not os.path.exists(insula_path):
        raise FileNotFoundError(f"Insula weights not found: {insula_path}")
    if not os.path.exists(beat_head_path):
        raise FileNotFoundError(f"Beat head weights not found: {beat_head_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load config to get best hyperparameters and architecture
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    best_peak_thr = config.get('best_peak_thr', 0.5)
    
    # Also load the main config.json to get architecture params
    main_config_path = os.path.join(os.path.dirname(checkpoint_dir), 'standalone_insula_module', 'config.json')
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r') as f:
            main_config = json.load(f)
        n_aINS = main_config['model_architecture']['n_aINS']
    else:
        # Fallback: infer from beat head weights
        beat_head_weights = torch.load(beat_head_path, map_location='cpu')
        if 'proj.weight' in beat_head_weights:
            n_aINS = beat_head_weights['proj.weight'].shape[1]
        else:
            n_aINS = beat_head_weights['weight'].shape[1]
    
    print(f"🔧 Loading trained model from: {checkpoint_dir}")
    print(f"   Best peak threshold: {best_peak_thr}")
    print(f"   n_aINS units: {n_aINS}")
    
    # Initialize modules (don't load pretrained weights)
    insula = InsulaModule(device=device, load_pretrained=False)
    beat_head = torch.nn.Linear(n_aINS, 1)  # Create beat head with correct dimensions
    beat_head.to(device)
    
    # Load weights
    insula_weights = torch.load(insula_path, map_location=device)
    beat_head_weights = torch.load(beat_head_path, map_location=device)
    
    # Handle beat head weight mapping (proj.weight/proj.bias -> weight/bias)
    if 'proj.weight' in beat_head_weights:
        beat_head_state_dict = {
            'weight': beat_head_weights['proj.weight'],
            'bias': beat_head_weights['proj.bias']
        }
    else:
        beat_head_state_dict = beat_head_weights
    
    insula.load_state_dict(insula_weights)
    beat_head.load_state_dict(beat_head_state_dict)
    
    insula.eval()
    beat_head.eval()
    
    print("✅ Trained model loaded successfully!")
    
    return insula, beat_head, best_peak_thr




def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=== Testing Standalone Insula Module ===\n")
    
    # Load ECG library
    ecg_list, meta = load_ecg_library(args.ecg_library_path, args.hrv_scale)
    
    # Create batch exactly like smoke_test.py
    rng = np.random.RandomState(args.seed)
    duration_s = rng.uniform(3.0, 8.0)
    T = int(round(duration_s * args.sampling_rate))
    batch_ecg = np.zeros((args.batch_size, T), dtype=np.float32)
    
    for b in range(args.batch_size):
        idx = rng.randint(0, len(ecg_list))
        ecg = ecg_list[idx].astype(np.float32)
        src_fs = int(meta[idx].get('sampling_rate', args.sampling_rate))
        src_len = len(ecg)
        need_len = int(round(duration_s * src_fs))
        
        if src_len >= need_len:
            start = 0 if src_len == need_len else rng.randint(0, src_len - need_len)
            segment = ecg[start:start + need_len]
        else:
            reps = int(np.ceil(need_len / max(1, src_len)))
            segment = np.tile(ecg, reps)[:need_len]
        
        resampled = resample_1d(segment, src_fs, args.sampling_rate, out_len=T)
        batch_ecg[b] = resampled
    
    print(f"Batch shape: {batch_ecg.shape}")
    
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    insula, beat_head, best_peak_thr = load_trained_model(args.checkpoint_dir, device)
    
    # Use the best peak threshold from training if not specified
    if args.peak_thr == 0.5:  # Default value, use trained threshold
        args.peak_thr = best_peak_thr
        print(f"🎯 Using trained peak threshold: {args.peak_thr:.3f}")
    
    # Create targets exactly like smoke_test.py
    targets, true_peaks_list = make_event_targets(
        batch_ecg, fs=args.sampling_rate, sigma_ms=args.sigma_ms, 
        zthr=args.zthr, refractory_ms=args.refractory_ms
    )
    
    # Process with standalone insula
    all_logits = []
    for b in range(args.batch_size):
        ecg_signal = batch_ecg[b]
        aINS_activity = insula.process_ecg(ecg_signal)  # [T, 10]
        
        with torch.no_grad():
            aINS_tensor = aINS_activity.to(insula.device)
            logits = beat_head(aINS_tensor)  # [T, 1]
            all_logits.append(logits)
    
    # Stack logits for metrics
    logits_bt = torch.stack(all_logits, dim=0).squeeze(-1)  # [B, T]
    
    # Compute metrics exactly like smoke_test.py
    probs_bt = torch.sigmoid(logits_bt).cpu().numpy()
    pred_peaks_list = []
    for b in range(args.batch_size):
        peaks = []
        probs = probs_bt[b]
        peak_min_dist_samples = int(args.peak_min_dist_ms * args.sampling_rate / 1000.0)
        
        for i in range(len(probs)):
            if (probs[i] > args.peak_thr and 
                (not peaks or i - peaks[-1] >= peak_min_dist_samples)):
                peaks.append(i)
        pred_peaks_list.append(np.array(peaks, dtype=np.int64))
    
    # Compute metrics
    tolerance_samples = int(args.tolerance_ms * args.sampling_rate / 1000)
    true_positives = 0
    total_pred = sum(len(peaks) for peaks in pred_peaks_list)
    total_true = sum(len(peaks) for peaks in true_peaks_list)
    
    for true_peaks, pred_peaks in zip(true_peaks_list, pred_peaks_list):
        for true_peak in true_peaks:
            for pred_peak in pred_peaks:
                if abs(true_peak - pred_peak) <= tolerance_samples:
                    true_positives += 1
                    break
    
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Results: F1={f1:.3f} P={precision:.3f} R={recall:.3f} TP={true_positives} Pred={total_pred} True={total_true}")
    
    # Simple plot
    rows = min(args.batch_size, 3)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
    if rows == 1:
        axes = np.array([axes])
    
    for i in range(rows):
        ecg = batch_ecg[i]
        true_peaks = true_peaks_list[i]
        pred_peaks = pred_peaks_list[i]
        t = np.arange(len(ecg)) / float(args.sampling_rate)
        
        # ECG + peaks
        ax = axes[i, 0]
        ax.plot(t, ecg, 'k-', lw=1)
        if true_peaks.size:
            ax.scatter(true_peaks / float(args.sampling_rate), ecg[true_peaks], c='g', s=18, label='True')
        if pred_peaks.size:
            ax.scatter(pred_peaks / float(args.sampling_rate), ecg[pred_peaks], c='r', s=12, marker='x', label='Pred')
        ax.set_title(f'ECG {i}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Targets vs Probs
        ax = axes[i, 1]
        ax.plot(t, targets[i], 'b-', lw=1.5, label='Target')
        ax.plot(t, probs_bt[i], 'm-', lw=1.0, alpha=0.8, label='Pred')
        ax.axhline(y=args.peak_thr, color='r', linestyle='--', alpha=0.7)
        ax.set_title(f'Target vs Prob {i}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    fig.suptitle(f"F1={f1:.3f} P={precision:.3f} R={recall:.3f}")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150)
    print(f"Saved plot to {args.save_path}")


if __name__ == "__main__":
    main()