#!/usr/bin/env python3
"""
Training script for standalone InsulaModule using best hyperparameters from config.json.

This script follows the training logic from insula_events but uses the standalone
insula module with configurable architecture and fixed connectivity.
"""
import os
import json
import argparse
import numpy as np
import torch
from torch import nn

from train_utils import ECGWindowLoader, make_event_targets, compute_metrics, set_seed
from insula_module import InsulaModule


class BeatHead(nn.Module):
    """Beat detection head for converting aINS activity to beat probabilities."""
    def __init__(self, in_dim: int = 10, bias_init: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)
        with torch.no_grad():
            if self.proj.bias is not None:
                self.proj.bias.fill_(float(bias_init))

    def forward(self, a_act_tba: torch.Tensor) -> torch.Tensor:
        # a_act_tba: [T, B, A]
        T, B, A = a_act_tba.shape
        logits_tb1 = self.proj(a_act_tba.reshape(T * B, A)).view(T, B, 1)
        return logits_tb1.permute(1, 0, 2).squeeze(2)  # [B, T]


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float, eps: float = 1e-6) -> torch.Tensor:
    """Focal loss implementation following insula_events logic."""
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    ce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
    pt = torch.where(targets > 0, p, 1 - p)
    w = (alpha * targets + (1 - alpha) * (1 - targets)) * ((1 - pt) ** gamma)
    return (w * ce).mean()


def eval_on_loader(eval_loader, insula, beat_head, args, n_batches=8):
    """Evaluate model on given loader following exact train_with_optuna_params logic."""
    insula_was_train = insula.training
    head_was_train = beat_head.training
    insula.eval()
    beat_head.eval()
    
    thr_candidates = (args.peak_thr, 0.8, 0.9)
    best_avg = None
    best_metrics = None
    best_thr_for_run = None
    
    with torch.no_grad():
        for thr in thr_candidates:
            f1s, precs, recs, ibis, dsdnns, drmssds = [], [], [], [], [], []
            for _ in range(n_batches):
                batch_ecg = eval_loader.next_batch(args.batch_size)
                hb = torch.from_numpy(batch_ecg).to(insula.device)
                
                aINS_tba = insula.process_ecg_batch(hb)  # [T, B, A]
                logits_bt = beat_head(aINS_tba)  # [B, T]
                
                m = compute_metrics(
                    logits_bt, hb,
                    fs=args.fs,
                    tolerance_ms=args.tolerance_ms,
                    peak_thr=float(thr),
                    peak_min_dist_ms=args.peak_min_dist_ms
                )
                f1s.append(float(m['f1']))
                precs.append(float(m['precision']))
                recs.append(float(m['recall']))
                ibis.append(float(m['ibi_mae']) if not (np.isnan(m['ibi_mae']) or np.isinf(m['ibi_mae'])) else float('inf'))
                dsdnns.append(float(m.get('d_sdnn', np.nan)))
                drmssds.append(float(m.get('d_rmssd', np.nan)))
            
            # Average across batches (filter inf IBI)
            f1_mean = float(np.mean(f1s))
            ibi_finite = [x for x in ibis if np.isfinite(x)]
            ibi_mean = float(np.mean(ibi_finite)) if ibi_finite else float('inf')
            # NaN-safe means for dSDNN and dRMSSD
            dsdnn_arr = np.array(dsdnns, dtype=float)
            dsdnn_mean = float(np.nanmean(dsdnn_arr)) if dsdnn_arr.size else float('nan')
            drmssd_arr = np.array(drmssds, dtype=float)
            drmssd_mean = float(np.nanmean(drmssd_arr)) if drmssd_arr.size else float('nan')
            avg = (f1_mean, -ibi_mean)
            if best_avg is None or avg > best_avg:
                best_avg = avg
                best_metrics = {
                    'f1': f1_mean,
                    'precision': float(np.mean(precs)),
                    'recall': float(np.mean(recs)),
                    'ibi_mae': ibi_mean,
                    'd_sdnn': dsdnn_mean,
                    'd_rmssd': drmssd_mean,
                }
                best_thr_for_run = float(thr)
    
    if insula_was_train:
        insula.train()
    if head_was_train:
        beat_head.train()
    
    return best_metrics, best_thr_for_run


def save_checkpoint(save_dir: str, insula: InsulaModule, beat_head: BeatHead, cfg: dict, tag: str = 'best'):
    """Save checkpoint in standalone module format."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save insula weights in module-native format
    insula_weights_path = os.path.join(save_dir, f'insula_weights_{tag}.pt')
    insula.save_weights(insula_weights_path)
    
    # Save beat head in compatible format
    head_weights_path = os.path.join(save_dir, f'beat_head_weights_{tag}.pt')
    state = beat_head.state_dict()
    payload = {
        'proj.weight': state['proj.weight'].detach().to('cpu'),
        'proj.bias': state['proj.bias'].detach().to('cpu'),
    }
    torch.save(payload, head_weights_path)
    
    # Save config
    with open(os.path.join(save_dir, f'config_{tag}.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description='Train standalone InsulaModule with best hyperparameters')
    
    # Required arguments (both train and val for validation-driven optimization)
    p.add_argument('--train_ecg_library_path', type=str, required=True, 
                   help='Path to training ECG library file (e.g., ecg_lib_train.npy)')
    p.add_argument('--val_ecg_library_path', type=str, required=True,
                   help='Path to validation ECG library file (e.g., ecg_lib_val.npy)')
    
    # Training parameters
    p.add_argument('--steps', type=int, default=3000, help='Training steps')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Other parameters
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--save_dir', type=str, default='runs/standalone_insula_train',
                   help='Directory to save results')
    p.add_argument('--log_every', type=int, default=100, help='Logging frequency')
    p.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return p.parse_args()


def main():
    """Main training function following insula_events logic."""
    args = parse_args()
    set_seed(args.seed)
    
    # Load config from standalone module
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters from config
    fs = config['ecg_processing']['fs']
    sigma_ms = 30.0  # Standard sigma for event targets
    zthr = config['peak_detection']['zthr']
    refractory_ms = config['peak_detection']['refractory_ms']
    tolerance_ms = config['peak_detection']['tolerance_ms']
    peak_min_dist_ms = config['peak_detection']['peak_min_dist_ms']
    peak_thr = config['peak_detection']['peak_thr']
    dt_ms = config['dynamics']['dt_ms']
    tau_ms = config['dynamics']['tau_ms']
    
    # Load best hyperparameters from config
    best_hyperparams = config.get('best_hyperparameters', {})
    focal_alpha = best_hyperparams.get('focal_alpha', 0.5)
    focal_gamma = best_hyperparams.get('focal_gamma', 1.5)
    sparsity_weight = best_hyperparams.get('sparsity_weight', 1e-3)
    beat_head_bias = best_hyperparams.get('beat_head_bias', -2.0)
    
    # Store in args for eval_on_loader compatibility
    args.fs = fs
    args.tolerance_ms = tolerance_ms
    args.peak_min_dist_ms = peak_min_dist_ms
    args.peak_thr = peak_thr
    
    print("=== Standalone Insula Training (Validation-Driven) ===")
    print(f"Train ECG library: {args.train_ecg_library_path}")
    print(f"Val ECG library: {args.val_ecg_library_path}")
    print(f"Steps: {args.steps}, Batch size: {args.batch_size}")
    print(f"Loss: focal (α={focal_alpha}, γ={focal_gamma})")
    print(f"Config: fs={fs}Hz, dt={dt_ms}ms, tau={tau_ms}ms")
    print(f"Hyperparams: sparsity_weight={sparsity_weight:.2e}, beat_head_bias={beat_head_bias}")
    print(f"Device: {args.device}")
    
    # Data loaders (following train_with_optuna_params pattern)
    rng = np.random.RandomState(args.seed)
    print(f"📁 Loading TRAIN ECG data from {args.train_ecg_library_path}...")
    train_loader = ECGWindowLoader(
        path=args.train_ecg_library_path,
        target_fs=fs,
        rng=rng,
        min_s=3.0,
        max_s=8.0,
    )
    print(f"✅ TRAIN ECG data loaded: {len(train_loader.ecg_list)} samples")
    
    print(f"📁 Loading VAL ECG data from {args.val_ecg_library_path}...")
    val_loader = ECGWindowLoader(
        path=args.val_ecg_library_path,
        target_fs=fs,
        rng=np.random.RandomState(args.seed + 999),
        min_s=3.0,
        max_s=8.0,
    )
    print(f"✅ VAL ECG data loaded: {len(val_loader.ecg_list)} samples")
    
    # Model components - start from scratch (load_pretrained=False)
    insula = InsulaModule(device=args.device, freeze=False, load_pretrained=False)
    beat_head = BeatHead(in_dim=insula.n_aINS, bias_init=beat_head_bias).to(insula.device)
    
    # Optimizer
    params = list(insula.parameters()) + list(beat_head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop following insula_events logic
    best_key = (-1.0, float('inf'))
    
    for step in range(1, args.steps + 1):
        # Get batch from TRAINING data
        batch_ecg = train_loader.next_batch(args.batch_size)  # [B, T] np
        hb = torch.from_numpy(batch_ecg).to(insula.device)  # [B, T]
        B, T = hb.shape
        
        # Create targets
        targets, _ = make_event_targets(
            hb,
            fs=fs,
            sigma_ms=sigma_ms,
            zthr=zthr,
            refractory_ms=refractory_ms,
        )
        
        # Forward pass through standalone insula
        aINS_tba = insula.process_ecg_batch(hb)  # [T, B, A]
        logits_bt = beat_head(aINS_tba)  # [B, T]
        
        # Compute loss using best hyperparameters
        loss_main = focal_loss_with_logits(logits_bt, targets, alpha=focal_alpha, gamma=focal_gamma)
        loss = loss_main + sparsity_weight * torch.sigmoid(logits_bt).mean()
        
        # Optimization step
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        
        # Apply structural mask to maintain connectivity constraints
        insula.apply_structural_mask()
        
        # Evaluation and logging following exact train_with_optuna_params logic
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                # First evaluate on current training batch
                thr_list = [0.2, 0.3, 0.4, 0.5]
                best_m, best_thr = None, None
                for thr in thr_list:
                    m = compute_metrics(
                        logits_bt, hb, 
                        fs=fs,
                        tolerance_ms=tolerance_ms,
                        peak_thr=thr, 
                        peak_min_dist_ms=peak_min_dist_ms
                    )
                    if best_m is None or (m['f1'], -(m['ibi_mae'] if not (np.isnan(m['ibi_mae']) or np.isinf(m['ibi_mae'])) else 1e9)) > \
                                         (best_m['f1'], -(best_m['ibi_mae'] if not (np.isnan(best_m['ibi_mae']) or np.isinf(best_m['ibi_mae'])) else 1e9)):
                        best_m, best_thr = m, thr
            
            # VALIDATION-DRIVEN: Replace training metrics with validation metrics
            best_thr_val = peak_thr
            metrics, best_thr_val = eval_on_loader(val_loader, insula, beat_head, args, n_batches=8)
            
            # Probability drift analysis (following insula_events)
            probs_bt = torch.sigmoid(logits_bt)
            p_mean_t = probs_bt.mean(dim=0)  # [T]
            p_mean_early = p_mean_t[:T//4].mean().item()
            p_mean_late = p_mean_t[3*T//4:].mean().item()
            p_mean_global = p_mean_t.mean().item()
            
            # Logging following train_with_optuna_params format
            pos = float(targets.sum().item())
            neg = float(targets.numel() - targets.sum().item())
            loss_info = f"focal(α={focal_alpha},γ={focal_gamma})"
            
            print(f"Step {step:5d} | loss {loss.item():.6f} | f1 {metrics['f1']:.3f} | P {metrics['precision']:.3f} | R {metrics['recall']:.3f} | "
                  f"IBI-MAE {metrics['ibi_mae']:.3f}s | dSDNN {metrics['d_sdnn']:.3f}s | dRMSSD {metrics['d_rmssd']:.3f}s | {loss_info} | "
                  f"#pos {pos:.0f} #neg {neg:.0f} | best_thr {best_thr_val:.1f} | p_early {p_mean_early:.3f} p_late {p_mean_late:.3f} p_global {p_mean_global:.3f}")
            
            # Save best checkpoint based on VALIDATION metrics
            f1 = metrics['f1']
            ibi_mae = metrics['ibi_mae'] if not (np.isnan(metrics['ibi_mae']) or np.isinf(metrics['ibi_mae'])) else float('inf')
            key = (f1, -ibi_mae)
            
            if key > best_key:
                best_key = key
                cfg = {
                    'fs': fs,
                    'sigma_ms': sigma_ms,
                    'tolerance_ms': tolerance_ms,
                    'zthr': zthr,
                    'refractory_ms': refractory_ms,
                    'peak_thr': peak_thr,
                    'best_peak_thr': best_thr_val,
                    'peak_min_dist_ms': peak_min_dist_ms,
                    'dt_ms': dt_ms,
                    'tau_ms': tau_ms,
                    'seed': args.seed,
                    'focal_alpha': focal_alpha,
                    'focal_gamma': focal_gamma,
                    'sparsity_weight': sparsity_weight,
                    'beat_head_bias': beat_head_bias,
                }
                save_checkpoint(args.save_dir, insula, beat_head, cfg, tag='best')
    
    # Save final checkpoint
    cfg_last = cfg.copy()
    save_checkpoint(args.save_dir, insula, beat_head, cfg_last, tag='last')
    
    print(f"\n✅ Training completed!")
    print(f"Best Validation F1: {best_key[0]:.4f}")
    print(f"Saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
