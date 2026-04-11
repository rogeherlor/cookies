# -*- coding: utf-8 -*-
"""
TLIO Training Script — Leave-One-Out CV on KITTI clean sequences.

Usage
-----
# Single LOO fold (train on all clean seqs except val-seq):
python train_tlio.py --mode loo --val-seq 01 --epochs 200 --output artifacts/tlio/

# Train on ALL clean sequences (for deployment):
python train_tlio.py --mode all --epochs 200 --output artifacts/tlio/

# Resume from checkpoint:
python train_tlio.py --mode loo --val-seq 01 --epochs 200 --resume artifacts/tlio/fold_01_ckpt.pt

Network
-------
ResNet1D from external/tlio/src/network/model_resnet.py.
  Input : (B, 6, W)  — [gyro_ga(3) | accel_ga_motion(3)] in gravity-aligned frame
  Output: (mean(B,3), logstd(B,3)) — displacement prediction + log-std

Loss
----
Two-phase training following the original TLIO paper:
  Phase 1 (MSE pre-training, first half of epochs):
      L_mse = || dp_gt - dp_pred ||²

  Phase 2 (NLL fine-tuning, second half):
      L_nll = 0.5 * (dp_gt - dp_pred)^T @ inv(Σ) @ (dp_gt - dp_pred)
            + 0.5 * log|Σ|
      where Σ = diag(exp(2 * logstd[i]))

Both dp_gt and dp_pred are in the gravity-aligned frame.

LOO Clean Sequences
-------------------
01, 04, 06, 07, 08, 09, 10  (sequences 00, 02, 05 have data gaps; 03 no raw data)

References
----------
Liu et al., "TLIO: Tight Learned Inertial Odometry", IEEE RA-L 2020
https://doi.org/10.1109/LRA.2020.3007421
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_TLIO_SRC  = _REPO_ROOT / 'external/tlio/src'
_SCRIPTS   = _REPO_ROOT / 'scripts/positioning/python'

for p in [str(_TLIO_SRC), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl
from tlio_dataset import build_windows, TARGET_HZ

# ── Constants ─────────────────────────────────────────────────────────────────
CLEAN_SEQS = ['01', '04', '06', '07', '08', '09', '10']

WINDOW_SECONDS = 1.0   # 1 s — matches original TLIO paper (200 Hz × 1 s = W=200)
STRIDE_SECONDS = 0.05  # 50 ms stride — matches original 20 Hz training sample rate


# ── Loss functions ────────────────────────────────────────────────────────────

def _augment_imu(imu_w, accel_range=0.2, gyro_range=0.05):
    """
    Add per-batch random bias offsets to simulate IMU sensor error.
    Matches the original TLIO paper's data augmentation (accel_bias_range=0.2,
    gyro_bias_range=0.05).
    """
    B = imu_w.shape[0]
    imu_w = imu_w.clone()
    imu_w[:, 0:3, :] += (torch.rand(B, 3, 1, device=imu_w.device) * 2 - 1) * gyro_range
    imu_w[:, 3:6, :] += (torch.rand(B, 3, 1, device=imu_w.device) * 2 - 1) * accel_range
    return imu_w


def mse_loss(dp_pred, dp_gt):
    return ((dp_pred - dp_gt) ** 2).mean()


def nll_loss(dp_pred, logstd, dp_gt):
    """
    Diagonal Gaussian NLL.
    Σ = diag(exp(2·logstd))  ← TLIO paper Eq.3
    L = 0.5 * Σ_i [(dp_gt_i - dp_pred_i)² / exp(2*logstd_i) + 2*logstd_i]
    """
    err   = dp_gt - dp_pred                         # (B, 3)
    inv_var = torch.exp(-2. * logstd)               # (B, 3)
    loss  = 0.5 * (err ** 2 * inv_var + 2. * logstd)
    return loss.mean()


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_sequence_windows(seq_id: str, sample_rate: float = 100.0,
                          dataset: str = 'kitti'):
    """
    Load a single sequence and return windowed tensors.
    Returns (imu_windows, dp_gt_ga) where dp_gt_ga are ground-truth
    displacements in the gravity-aligned frame.
    """
    if dataset == 'cookies':
        nav = dl.get_cookies_dataset_by_id(seq_id, sample_rate=sample_rate)
    else:
        nav = dl.get_kitti_dataset(seq_id, sample_rate=sample_rate)
    # Window and stride in TARGET_HZ (200 Hz) samples — data is upsampled in build_windows
    W = int(WINDOW_SECONDS * TARGET_HZ)
    S = int(STRIDE_SECONDS * TARGET_HZ)

    imu_wins, dp_gt_enu, Rz_list, _ = build_windows(nav, W, S, target_rate=TARGET_HZ)

    # Rotate ground-truth displacement to gravity-aligned frame (Rz.T @ dp_enu)
    dp_gt_ga = np.stack([
        Rz_list[k].T @ dp_gt_enu[k] for k in range(len(Rz_list))
    ], axis=0)   # (M, 3)

    return (
        torch.from_numpy(imu_wins).float(),          # (M, 6, W)
        torch.from_numpy(dp_gt_ga).float(),          # (M, 3)
    )


def build_dataset(train_seqs: list, val_seq: str = None, sample_rate: float = 100.0,
                  dataset: str = 'kitti'):
    """Return (train_dataset, val_dataset or None)."""
    train_imu, train_dp = [], []
    for seq in train_seqs:
        print(f"  Loading seq {seq} ...", flush=True)
        try:
            imu_w, dp_w = load_sequence_windows(seq, sample_rate, dataset)
            train_imu.append(imu_w)
            train_dp.append(dp_w)
        except Exception as e:
            print(f"  WARNING: failed to load seq {seq}: {e}")

    train_dataset = TensorDataset(
        torch.cat(train_imu, dim=0),
        torch.cat(train_dp,  dim=0),
    )

    val_dataset = None
    if val_seq is not None:
        print(f"  Loading val seq {val_seq} ...", flush=True)
        try:
            val_imu, val_dp = load_sequence_windows(val_seq, sample_rate, dataset)
            val_dataset = TensorDataset(val_imu, val_dp)
        except Exception as e:
            print(f"  WARNING: failed to load val seq {val_seq}: {e}")

    return train_dataset, val_dataset


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    import importlib.util
    _mf_path = _HERE / 'network/model_factory.py'
    _spec = importlib.util.spec_from_file_location('_tlio_model_factory', _mf_path)
    _mf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mf)
    get_model = _mf.get_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # Resolve the active sequence list based on dataset
    if args.dataset == 'cookies':
        from data_loader import COOKIES_CLEAN_SEQS
        CLEAN_SEQS_ACTIVE = list(COOKIES_CLEAN_SEQS.keys())   # ['c01'..'c06']
    else:
        CLEAN_SEQS_ACTIVE = CLEAN_SEQS   # kitti clean seqs

    # Determine sequences
    if args.mode == 'loo':
        if args.val_seq not in CLEAN_SEQS_ACTIVE:
            raise ValueError(f"--val-seq must be one of {CLEAN_SEQS_ACTIVE}")
        train_seqs = [s for s in CLEAN_SEQS_ACTIVE if s != args.val_seq]
        val_seq    = args.val_seq
        out_name   = f'fold_{args.val_seq}.pt'
    else:  # all
        train_seqs = CLEAN_SEQS_ACTIVE
        val_seq    = None
        out_name   = 'tlio_resnet.pt'

    print(f"Dataset={args.dataset}  Mode={args.mode}  train={train_seqs}  val={val_seq}")

    # Window size in TARGET_HZ (200 Hz) samples — same as build_windows
    sample_rate = 100.0
    W = int(WINDOW_SECONDS * TARGET_HZ)

    print("Building dataset ...")
    train_ds, val_ds = build_dataset(train_seqs, val_seq, sample_rate, args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True) if val_ds else None

    print(f"Train samples: {len(train_ds)},  "
          f"Val samples: {len(val_ds) if val_ds else 0}")

    # Model
    in_dim = W // 32 + 1
    model = get_model('resnet', {'in_dim': in_dim}, input_dim=6, output_dim=3)
    model = model.to(device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    optimizer  = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # LR warmup: 1% → 100% of lr over 5 epochs, then cosine decay to 1e-6
    _warmup_epochs = 5
    _warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                       total_iters=_warmup_epochs)
    _cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - _warmup_epochs, 1),
                                eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[_warmup, _cosine],
                             milestones=[_warmup_epochs])

    phase_switch = args.mse_epochs    # MSE pre-training for fixed N epochs, then NLL

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / out_name.replace('.pt', '_ckpt.pt')

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # ── Training ───────────────────────────────────────────────────
        model.train()
        total_loss = 0.
        for imu_w, dp_target in train_loader:
            imu_w     = _augment_imu(imu_w.to(device))
            dp_target = dp_target.to(device)

            optimizer.zero_grad()
            mean_pred, logstd_pred = model(imu_w)

            if epoch < phase_switch:
                loss = mse_loss(mean_pred, dp_target)
            else:
                loss = nll_loss(mean_pred, logstd_pred, dp_target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            total_loss += loss.item() * imu_w.size(0)

        scheduler.step()
        avg_train = total_loss / len(train_ds)
        phase = 'MSE' if epoch < phase_switch else 'NLL'
        # Print every epoch for first 50, then every 10th, always print last
        _should_print = (epoch < 50) or ((epoch + 1) % 10 == 0) or (epoch == args.epochs - 1)

        # ── Validation ─────────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss = 0.
            with torch.no_grad():
                for imu_w, dp_target in val_loader:
                    imu_w     = imu_w.to(device)
                    dp_target = dp_target.to(device)
                    mean_pred, logstd_pred = model(imu_w)
                    if epoch < phase_switch:
                        loss = mse_loss(mean_pred, dp_target)
                    else:
                        loss = nll_loss(mean_pred, logstd_pred, dp_target)
                    val_loss += loss.item() * imu_w.size(0)
            avg_val = val_loss / len(val_ds)
            if _should_print:
                print(f"Epoch {epoch+1:4d}/{args.epochs} [{phase}]  "
                      f"train={avg_train:.4f}  val={avg_val:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss':         avg_val,
                    'args':             vars(args),
                }, output_dir / out_name)
                print(f"  → saved best weights ({out_name})")
        else:
            if _should_print:
                print(f"Epoch {epoch+1:4d}/{args.epochs} [{phase}]  "
                      f"train={avg_train:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Periodic checkpoint (every 20 epochs)
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'args':             vars(args),
            }, ckpt_path)

    # If no validation, save at the end
    if val_loader is None:
        torch.save({
            'epoch':            args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'args':             vars(args),
        }, output_dir / out_name)
        print(f"Saved final weights → {output_dir / out_name}")

    # ── Convergence diagnostics ────────────────────────────────────────────
    # A converged model should: (a) show varying dp_pred across windows (not constant),
    # (b) show logstd that varies per sample (not all identical).
    print("Computing convergence diagnostics ...")
    model.eval()
    all_dp, all_logstd = [], []
    with torch.no_grad():
        for imu_w, _ in train_loader:
            mean_pred, logstd_pred = model(imu_w.to(device))
            all_dp.append(mean_pred.cpu().numpy())
            all_logstd.append(logstd_pred.cpu().numpy())
    all_dp     = np.concatenate(all_dp,     axis=0)
    all_logstd = np.concatenate(all_logstd, axis=0)
    print(f"  dp_pred  mean  (m) : {all_dp.mean(axis=0)}")
    print(f"  dp_pred  std   (m) : {all_dp.std(axis=0)}")
    print(f"  logstd   mean      : {all_logstd.mean(axis=0)}")
    print(f"  logstd   std       : {all_logstd.std(axis=0)}")
    print(f"  exp(logstd) mean   : {np.exp(all_logstd).mean(axis=0)}")

    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="TLIO ResNet training")
    parser.add_argument('--dataset',    choices=['kitti', 'cookies'], default='kitti',
                        help="Dataset family (default: kitti)")
    parser.add_argument('--mode',       choices=['loo', 'all'], default='loo',
                        help="'loo' = LOO fold (requires --val-seq), 'all' = all clean seqs")
    parser.add_argument('--val-seq',    default='01',
                        help="Validation sequence for LOO (kitti: '01', cookies: 'c01', …)")
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--mse-epochs', type=int, default=20,
                        help='Number of MSE pre-training epochs before switching to NLL '
                             '(default: 20 — matches original TLIO paper epoch 10, '
                             'remaining epochs train calibrated uncertainty)')
    parser.add_argument('--output',     default=str(_REPO_ROOT / 'artifacts/tlio'),
                        help="Directory to save model weights")
    parser.add_argument('--resume',     default=None,
                        help="Path to checkpoint .pt to resume training from")
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
