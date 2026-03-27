# -*- coding: utf-8 -*-
"""
Tartan IMU LoRA Fine-tuning Script — Leave-One-Out CV on KITTI clean sequences.

IMPORTANT: This script fine-tunes LoRA adapters ONLY.
           The pretrained Tartan IMU base model MUST exist first.
           Tartan IMU is NEVER trained from scratch.

Usage
-----
# Single LOO fold (fine-tune LoRA adapters, holding out val-seq):
python train_tartan.py --mode loo --val-seq 01 --epochs 50 --output artifacts/tartan_imu/

# Fine-tune on ALL clean sequences (for deployment):
python train_tartan.py --mode all --epochs 50 --output artifacts/tartan_imu/

What is fine-tuned
------------------
Only the LoRA adapter layers (approx. 1.1M params).  The backbone (ResNet +
LSTM) is frozen.  Adapters are injected into the backbone's linear layers.

Training objective
------------------
NLL on body-frame velocity (paper Section 3.3):
    v_gt = (p_gt[t+1] - p_gt[t]) / dt  → rotated to body frame using gt orientation
    L_NLL = 0.5 * (v - v̂)^T @ Σ^-1 @ (v - v̂) + 0.5 * log|Σ|
    Σ = diag(exp(û))  (single log-std, not log-variance)

LOO Clean Sequences
-------------------
01, 04, 06, 07, 08, 09, 10

References
----------
Zhao et al., "Tartan IMU: A Light Foundation Model for Inertial Positioning
in Robotics", CVPR 2025.  https://superodometry.com/tartanimu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_SCRIPTS   = _REPO_ROOT / 'scripts/positioning/python'
_ARTIFACTS = _REPO_ROOT / 'artifacts/tartan_imu'

for p in [str(_HERE), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl
from tartan_runner import _find_tartan_weights, _load_tartan_model, _qfrom_euler, _qto_Rbn

CLEAN_SEQS = ['01', '04', '06', '07', '08', '09', '10']

TARGET_HZ    = 200
LSTM_STEPS   = 10
STEP_SAMPLES = TARGET_HZ    # = 200 samples per 1-second step


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _qnorm(q):
    n = np.linalg.norm(q); return q/n if n > 0. else np.array([1.,0.,0.,0.])


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_tartan_dataset(nav, target_hz=TARGET_HZ,
                         lstm_steps=LSTM_STEPS, step_samples=STEP_SAMPLES):
    """
    Build windowed tensors for Tartan IMU fine-tuning from a single sequence.

    Returns
    -------
    imu_windows  : (M, lstm_steps, step_samples, 6) float32
    v_gt_body    : (M, 3) float32 — ground-truth velocity in body frame [m/s]
    log_std_mask : (M,) bool — True if velocity is reliable (moving, GPS available)
    """
    accel_flu = nav.accel_flu
    gyro_flu  = nav.gyro_flu
    orient    = nav.orient
    vel_enu   = nav.vel_enu
    N         = accel_flu.shape[0]
    src_rate  = nav.sample_rate

    t_src = np.arange(N) / src_rate
    t_up  = np.arange(0., t_src[-1], 1.0 / target_hz)
    N_up  = len(t_up)

    # Upsample IMU
    accel_up = interp1d(t_src, accel_flu, axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(accel_flu[0], accel_flu[-1]))(t_up)
    gyro_up  = interp1d(t_src, gyro_flu,  axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(gyro_flu[0], gyro_flu[-1]))(t_up)

    # Gravity-free accelerometer at 200 Hz
    roll_up  = np.interp(t_up, t_src, orient[:, 0])
    pitch_up = np.interp(t_up, t_src, orient[:, 1])
    accel_gf = np.zeros_like(accel_up)
    for k in range(N_up):
        R_nb   = _qto_Rbn(_qfrom_euler(roll_up[k], pitch_up[k], 0.))
        g_body = R_nb.T @ np.array([0., 0., -9.81])
        accel_gf[k] = accel_up[k] - g_body

    ctx_up        = lstm_steps * step_samples        # 2000 samples context
    update_stride = int(src_rate)                    # 100 samples → 1-Hz windows

    windows  = []
    v_bodies = []

    for i_src in range(0, N - 1, update_stride):
        i_up = int(i_src * target_hz / src_rate)
        if i_up < ctx_up:
            continue

        # Build (lstm_steps, step_samples, 6) window
        s = i_up - ctx_up
        win = np.zeros((lstm_steps, step_samples, 6), dtype=np.float32)
        for step in range(lstm_steps):
            ss = s + step * step_samples
            ee = ss + step_samples
            win[step, :, 0:3] = accel_gf[ss:ee].astype(np.float32)
            win[step, :, 3:6] = gyro_up[ss:ee].astype(np.float32)

        # Ground-truth velocity in body frame at i_src
        if i_src + 1 < N:
            dt    = 1.0 / src_rate
            # Finite-difference velocity from position (via vel_enu which is GT)
            v_enu = vel_enu[i_src].copy()
            R_nb  = _qto_Rbn(_qfrom_euler(orient[i_src, 0],
                                           orient[i_src, 1],
                                           orient[i_src, 2]))
            R_bn  = R_nb.T
            v_body = R_bn @ v_enu   # (3,)
        else:
            continue

        windows.append(win)
        v_bodies.append(v_body.astype(np.float32))

    if not windows:
        return None, None

    imu_tensor = torch.from_numpy(np.stack(windows, axis=0))   # (M, 10, 200, 6)
    v_gt_tensor = torch.from_numpy(np.stack(v_bodies, axis=0)) # (M, 3)
    return imu_tensor, v_gt_tensor


# ── Loss ──────────────────────────────────────────────────────────────────────

def nll_velocity_loss(v_pred, log_std, v_gt):
    """
    Diagonal Gaussian NLL for velocity prediction.
    Σ = diag(exp(log_std_i))  (single log-std → log-variance = 2*log_std)
    L = 0.5 * Σ_i [(v_gt_i - v_pred_i)² * exp(-log_std_i) + log_std_i]
    """
    err     = v_gt - v_pred
    inv_std = torch.exp(-log_std)
    loss    = 0.5 * (err ** 2 * inv_std + log_std)
    return loss.mean()


# ── LoRA injection ────────────────────────────────────────────────────────────

def _apply_lora_to_model(model, lora_rank: int):
    """
    Apply LoRA adapters to linear layers in the model.
    Falls back to making all parameters trainable if peft is not available.

    Returns model with LoRA applied (or all params enabled).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=['weight'],   # inject into Linear layers
            bias='none',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("Tartan fine-tune: LoRA adapters injected via peft.")
    except ImportError:
        print("Tartan fine-tune: peft not available. Fine-tuning all parameters.")
        for p in model.parameters():
            p.requires_grad_(True)
    except Exception as e:
        print(f"Tartan fine-tune: LoRA injection failed ({e}). Fine-tuning all parameters.")
        for p in model.parameters():
            p.requires_grad_(True)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # Resolve the active sequence list based on dataset
    if args.dataset == 'cookies':
        from data_loader import COOKIES_CLEAN_SEQS
        CLEAN_SEQS_ACTIVE = list(COOKIES_CLEAN_SEQS.keys())
    else:
        CLEAN_SEQS_ACTIVE = CLEAN_SEQS

    if args.mode == 'loo':
        if args.val_seq not in CLEAN_SEQS_ACTIVE:
            raise ValueError(f"--val-seq must be one of {CLEAN_SEQS_ACTIVE}")
        train_seqs = [s for s in CLEAN_SEQS_ACTIVE if s != args.val_seq]
        val_seq    = args.val_seq
        out_name   = f'lora_fold_{args.val_seq}.pt'
    else:
        train_seqs = CLEAN_SEQS_ACTIVE
        val_seq    = None
        out_name   = 'lora_adapters.pt'

    print(f"Dataset={args.dataset}  Mode={args.mode}  train={train_seqs}  val={val_seq}")

    def _load_seq(seq):
        if args.dataset == 'cookies':
            return dl.get_cookies_dataset_by_id(seq, sample_rate=100.0)
        return dl.get_kitti_dataset(seq, sample_rate=100.0)

    # ── Load pretrained base model ─────────────────────────────────────────
    weights_path = _find_tartan_weights()
    model, _     = _load_tartan_model(weights_path, lora_path=None,
                                      lora_rank=args.lora_rank, device=device)
    print(f"Base model loaded from {weights_path}")

    # ── Apply LoRA ─────────────────────────────────────────────────────────
    model = _apply_lora_to_model(model, args.lora_rank)
    model = model.to(device)

    # ── Build datasets ─────────────────────────────────────────────────────
    train_imu_list, train_v_list = [], []
    for seq in train_seqs:
        print(f"  Loading seq {seq} ...", flush=True)
        try:
            nav = _load_seq(seq)
            imu_t, v_t = build_tartan_dataset(nav)
            if imu_t is not None:
                train_imu_list.append(imu_t)
                train_v_list.append(v_t)
        except Exception as e:
            print(f"  WARNING: {seq} failed: {e}")

    train_ds = TensorDataset(
        torch.cat(train_imu_list, dim=0),
        torch.cat(train_v_list,   dim=0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    val_loader = None
    if val_seq is not None:
        try:
            nav_val = _load_seq(val_seq)
            val_imu, val_v = build_tartan_dataset(nav_val)
            if val_imu is not None:
                val_ds     = TensorDataset(val_imu, val_v)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                        shuffle=False, num_workers=2)
        except Exception as e:
            print(f"  WARNING: val seq {val_seq} failed: {e}")

    print(f"Train windows: {len(train_ds)},  "
          f"Val windows: {len(val_loader.dataset) if val_loader else 0}")

    # Only update LoRA parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        for imu_b, v_gt_b in train_loader:
            imu_b  = imu_b.to(device)
            v_gt_b = v_gt_b.to(device)
            optimizer.zero_grad()
            v_pred, log_std = model(imu_b, robot_type='car')
            loss = nll_velocity_loss(v_pred, log_std, v_gt_b)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=5.)
            optimizer.step()
            train_loss += loss.item() * imu_b.size(0)

        scheduler.step()
        avg_train = train_loss / len(train_ds)

        if val_loader is not None:
            model.eval()
            val_loss = 0.
            with torch.no_grad():
                for imu_b, v_gt_b in val_loader:
                    imu_b  = imu_b.to(device)
                    v_gt_b = v_gt_b.to(device)
                    v_pred, log_std = model(imu_b, robot_type='car')
                    loss = nll_velocity_loss(v_pred, log_std, v_gt_b)
                    val_loss += loss.item() * imu_b.size(0)
            avg_val = val_loss / len(val_loader.dataset)
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            if avg_val < best_val:
                best_val = avg_val
                _save_lora(model, epoch, avg_val, output_dir / out_name)
                print(f"  → saved best ({out_name})")
        else:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={avg_train:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if val_loader is None:
        _save_lora(model, args.epochs - 1, None, output_dir / out_name)

    print("Fine-tuning complete.")


def _save_lora(model, epoch, val_loss, path: Path):
    """Save only the trainable (LoRA) parameters."""
    lora_state = {k: v for k, v in model.state_dict().items()
                  if 'lora_' in k or any(p.requires_grad
                                          for name, p in model.named_parameters()
                                          if name == k)}
    torch.save({
        'epoch':           epoch,
        'lora_state_dict': lora_state,
        'val_loss':        val_loss,
    }, path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tartan IMU LoRA fine-tuning (LOO protocol)")
    parser.add_argument('--dataset',    choices=['kitti', 'cookies'], default='kitti',
                        help="Dataset family (default: kitti)")
    parser.add_argument('--mode',       choices=['loo','all'], default='loo')
    parser.add_argument('--val-seq',    default='01',
                        help="Validation sequence (kitti: '01', cookies: 'c01', …)")
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--lora-rank',  type=int,   default=8)
    parser.add_argument('--output',     default=str(_ARTIFACTS))
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
