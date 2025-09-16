import os
import math
from typing import List, Dict, Tuple
import numpy as np
import torch


# From data.py

class ECGWindowLoader:
    """Loads ECG library and yields random windows as batches [B, T]."""
    def __init__(self, path: str, target_fs: int, rng: np.random.RandomState,
                 min_s: float = 3.0, max_s: float = 8.0, hrv_scale: float = None, verbose: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"ECG library not found: {path}")
        
        self.verbose = verbose
        if self.verbose:
            print(f"🔍 ECGWindowLoader debugging:")
            print(f"  Loading from: {path}")
        
        lib = np.load(path, allow_pickle=True).item()
        all_ecg_list: List[np.ndarray] = list(lib.get('ecg_samples', []))
        all_meta: List[Dict] = list(lib.get('metadata', [{} for _ in all_ecg_list]))
        
        print(f"  Total ECG samples loaded: {len(all_ecg_list)}")
        if len(all_ecg_list) > 0:
            first_ecg = all_ecg_list[0]
            print(f"  First ECG sample: shape={first_ecg.shape}, dtype={first_ecg.dtype}")
            print(f"  First ECG sample min/max/mean: {first_ecg.min():.6f} / {first_ecg.max():.6f} / {first_ecg.mean():.6f}")
            print(f"  First ECG sample first 10 values: {first_ecg[:10]}")
        
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
            
            self.ecg_list = [all_ecg_list[i] for i in filtered_indices]
            self.meta = [all_meta[i] for i in filtered_indices]
            print(f"  Filtered to {len(self.ecg_list)} samples with hrv_scale={hrv_scale}")
        else:
            self.ecg_list = all_ecg_list
            self.meta = all_meta
            print(f"  Using all {len(self.ecg_list)} samples (no HRV filtering)")
        
        self.src_fs_list: List[int] = []
        for m in self.meta:
            fs = int(m.get('sampling_rate', target_fs))
            self.src_fs_list.append(fs)
        self.target_fs = int(target_fs)
        self.rng = rng
        self.min_s = float(min_s)
        self.max_s = float(max_s)

    def next_batch(self, batch_size: int) -> np.ndarray:
        duration_s = self.rng.uniform(self.min_s, self.max_s)
        T = int(round(duration_s * self.target_fs))
        batch = np.zeros((batch_size, T), dtype=np.float32)
        
        #print(f"🔍 next_batch debugging:")
        #print(f"  batch_size: {batch_size}, duration_s: {duration_s:.3f}, T: {T}")
        #print(f"  Available ECG samples: {len(self.ecg_list)}")
        
        for b in range(batch_size):
            idx = self.rng.randint(0, len(self.ecg_list))
            ecg = self.ecg_list[idx].astype(np.float32)
            src_fs = self.src_fs_list[idx]
            src_len = len(ecg)
            need_len = int(round(duration_s * src_fs))
            
            # print(f"  Sample {b}: idx={idx}, src_fs={src_fs}, src_len={src_len}, need_len={need_len}")
            # print(f"    Original ECG min/max/mean: {ecg.min():.6f} / {ecg.max():.6f} / {ecg.mean():.6f}")
            
            if src_len >= need_len:
                start = 0 if src_len == need_len else self.rng.randint(0, src_len - need_len)
                segment = ecg[start:start + need_len]
            else:
                reps = int(math.ceil(need_len / max(1, src_len)))
                segment = np.tile(ecg, reps)[:need_len]
            
            # print(f"    Segment min/max/mean: {segment.min():.6f} / {segment.max():.6f} / {segment.mean():.6f}")
            
            resampled = resample_1d(segment, src_fs, self.target_fs, out_len=T)
            # print(f"    Resampled min/max/mean: {resampled.min():.6f} / {resampled.max():.6f} / {resampled.mean():.6f}")
            # print(f"    Resampled first 5 values: {resampled[:5]}")
            
            batch[b] = resampled
        
        # print(f"  Final batch min/max/mean: {batch.min():.6f} / {batch.max():.6f} / {batch.mean():.6f}")
        return batch

# From labels.py

def detect_rpeaks_simple(ecg: np.ndarray, fs: int, zthr: float = 2.5, refractory_ms: int = 300) -> np.ndarray:
    if len(ecg) < 3:
        return np.array([], dtype=np.int64)
    z = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
    refractory = max(1, int(round(refractory_ms * fs / 1000.0)))
    peaks: List[int] = []
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


def build_gaussian_kernel_1d(sigma_samples: float, max_sigma_mult: int = 6) -> torch.Tensor:
    sigma = max(1e-6, float(sigma_samples))
    radius = int(max(1, round(max_sigma_mult * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    # Peak normalize so center value is 1.0 (training target near peaks is 1.0)
    kernel = kernel / kernel.max().clamp(min=1e-8)
    return kernel.view(1, 1, -1)


def make_event_targets(batch_ecg: torch.Tensor, fs: int, sigma_ms: float,
                       zthr: float = 2.5, refractory_ms: int = 300) -> Tuple[torch.Tensor, List[np.ndarray]]:
    device = batch_ecg.device
    B, T = batch_ecg.shape
    all_peaks: List[np.ndarray] = []
    base = torch.zeros((B, T), dtype=torch.float32)
    ecg_np = batch_ecg.detach().cpu().numpy()
    for b in range(B):
        peaks = detect_rpeaks_simple(ecg_np[b], fs=fs, zthr=zthr, refractory_ms=refractory_ms)
        all_peaks.append(peaks)
        if peaks.size:
            peaks_clamped = peaks[(peaks >= 0) & (peaks < T)]
            base[b, peaks_clamped] = 1.0
    base = base.to(device)
    sigma_samples = (sigma_ms / 1000.0) * fs
    kernel = build_gaussian_kernel_1d(sigma_samples).to(device)
    x = base.unsqueeze(1)
    padding = (kernel.shape[-1] - 1) // 2
    smoothed = torch.nn.functional.conv1d(x, kernel, padding=padding)
    smoothed = torch.clamp(smoothed.squeeze(1), 0.0, 1.0)
    return smoothed, all_peaks

# From metrics.py

def greedy_peak_match(pred: np.ndarray, true: np.ndarray, tol_samples: int) -> Tuple[int, int, int]:
    pred = list(pred.astype(int))
    true = list(true.astype(int))
    tp = 0
    used_true = set()
    for p in pred:
        best_j = None
        best_dist = tol_samples + 1
        for j, t in enumerate(true):
            if j in used_true:
                continue
            d = abs(p - t)
            if d <= tol_samples and d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            tp += 1
            used_true.add(best_j)
    fp = len(pred) - tp
    fn = len(true) - tp
    return tp, fp, fn


def ibi_stats(pred_idx: np.ndarray, true_idx: np.ndarray, fs: int) -> Tuple[float, float, float]:
    if len(pred_idx) < 2 or len(true_idx) < 2:
        return float('nan'), float('nan'), float('nan')
    pred_ibi = np.diff(pred_idx) / float(fs)
    true_ibi = np.diff(true_idx) / float(fs)
    n = min(len(pred_ibi), len(true_ibi))
    pred_ibi = pred_ibi[:n]
    true_ibi = true_ibi[:n]
    mae = float(np.mean(np.abs(pred_ibi - true_ibi)))
    d_sdnn = float(abs(np.std(pred_ibi) - np.std(true_ibi)))
    d_rmssd = float(abs(rmssd(pred_ibi) - rmssd(true_ibi)))
    return mae, d_sdnn, d_rmssd


def peaks_from_probs(probs_bt: np.ndarray, fs: int, peak_thr: float = 0.5, peak_min_dist_ms: float = 300.0) -> List[np.ndarray]:
    B, T = probs_bt.shape
    results: List[np.ndarray] = []
    min_dist = int(round((peak_min_dist_ms / 1000.0) * fs))
    thr = float(peak_thr)
    for b in range(B):
        x = probs_bt[b]
        idxs: List[int] = []
        i = 1
        while i < T - 1:
            if x[i] >= thr and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
                if not idxs or (i - idxs[-1]) >= min_dist:
                    idxs.append(i)
                    i += min_dist
                    continue
            i += 1
        results.append(np.array(idxs, dtype=np.int64))
    return results


def compute_metrics(logits_bt: torch.Tensor, batch_ecg_bt: torch.Tensor, fs: int, tolerance_ms: int,
                   peak_thr: float = 0.5, peak_min_dist_ms: float = 300.0
                   ) -> Dict[str, float]:
    with torch.no_grad():
        probs_bt = torch.sigmoid(logits_bt).detach().cpu().numpy()
        B, T = probs_bt.shape
        pred_peaks = peaks_from_probs(probs_bt, fs, peak_thr=peak_thr, peak_min_dist_ms=peak_min_dist_ms)
        ecg_np = batch_ecg_bt.detach().cpu().numpy()
        true_peaks = [detect_rpeaks_simple(ecg_np[b], fs) for b in range(B)]
        tol_samples = int(round(tolerance_ms * fs / 1000.0))

        tps = fps = fns = 0
        ibis = []
        dsdnn = []
        drmssd = []
        for b in range(B):
            tp, fp, fn = greedy_peak_match(pred_peaks[b], true_peaks[b], tol_samples)
            tps += tp; fps += fp; fns += fn
            mae, d_sd, d_rm = ibi_stats(pred_peaks[b], true_peaks[b], fs)
            if not (np.isnan(mae) or np.isinf(mae)):
                ibis.append(mae)
            if not (np.isnan(d_sd) or np.isinf(d_sd)):
                dsdnn.append(d_sd)
            if not (np.isnan(d_rm) or np.isinf(d_rm)):
                drmssd.append(d_rm)

        precision = tps / (tps + fps + 1e-8)
        recall = tps / (tps + fns + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'ibi_mae': float(np.mean(ibis)) if ibis else float('nan'),
            'd_sdnn': float(np.mean(dsdnn)) if dsdnn else float('nan'),
            'd_rmssd': float(np.mean(drmssd)) if drmssd else float('nan'),
        }
        return metrics

# From utils.py

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resample_1d(signal: np.ndarray, src_fs: int, dst_fs: int, out_len: int = None) -> np.ndarray:
    if src_fs == dst_fs and (out_len is None or out_len == len(signal)):
        return signal.copy()
    t_src = np.arange(len(signal)) / float(src_fs)
    if out_len is None:
        duration = t_src[-1] if len(t_src) > 1 else 0
        out_len = max(1, int(round(duration * dst_fs)))
    t_dst = np.arange(out_len) / float(dst_fs)
    return np.interp(t_dst, t_src, signal).astype(np.float32)


def rmssd(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    return float(np.sqrt(np.mean(diffs ** 2)))



