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
    v_gt_body    : (M, lstm_steps, 3) float32 — ground-truth body-frame velocity
                   per LSTM window (paper Eq. 2): the window-integrated relative
                   velocity v_{j→j+1} = Δp/dt rotated into the body frame, one
                   target per 1-second window rather than a single last-window one.
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

        # Per-window ground-truth body-frame velocity (paper Eq. 2).
        # Window w spans source indices [s_w, e_w) (1 s).  The target is the
        # window-integrated velocity  v_{j→j+1} = Δp/dt ≈ mean(vel_enu) over the
        # window, rotated into the body frame at the window end — not the
        # instantaneous vel_enu[i_src].  This matches what the pretrained backbone
        # regresses and supervises all 10 windows, not only the last.
        step_src = int(src_rate)   # source samples per 1-s window
        v_win    = np.zeros((lstm_steps, 3), dtype=np.float32)
        ok       = True
        for w in range(lstm_steps):
            s_w = i_src - (lstm_steps - w) * step_src
            e_w = i_src - (lstm_steps - 1 - w) * step_src
            if s_w < 0 or e_w > N:
                ok = False
                break
            v_enu_mean = vel_enu[s_w:e_w].mean(axis=0)              # Δp/dt over window
            io   = min(e_w, N - 1)
            R_nb = _qto_Rbn(_qfrom_euler(orient[io, 0], orient[io, 1], orient[io, 2]))
            v_win[w] = (R_nb.T @ v_enu_mean).astype(np.float32)     # body-frame velocity
        if not ok:
            continue

        windows.append(win)
        v_bodies.append(v_win)

    if not windows:
        return None, None

    imu_tensor  = torch.from_numpy(np.stack(windows, axis=0))   # (M, 10, 200, 6)
    v_gt_tensor = torch.from_numpy(np.stack(v_bodies, axis=0))  # (M, 10, 3)
    return imu_tensor, v_gt_tensor


# ── Loss ──────────────────────────────────────────────────────────────────────

def nll_velocity_loss(v_pred, log_std, v_gt):
    """
    Diagonal Gaussian NLL for velocity prediction.
    Σ = diag(exp(log_std_i))  (single log-std → log-variance = 2*log_std)
    L = 0.5 * Σ_i [(v_gt_i - v_pred_i)² * exp(-log_std_i) + log_std_i]

    Shapes broadcast over any leading dims, so this handles both a single
    prediction (B, 3) and the per-window sequence (B, lstm_steps, 3) used for
    the paper's Eq. 2 multi-window supervision; .mean() averages over windows.
    """
    err     = v_gt - v_pred
    inv_std = torch.exp(-log_std)
    loss    = 0.5 * (err ** 2 * inv_std + log_std)
    return loss.mean()


# ── LoRA injection ────────────────────────────────────────────────────────────

# Backbone Linear layers to adapt with LoRA: the transformer trunk's attention
# output projection and MLP. The Conv/LSTM backbone and the pretrained per-robot
# heads stay frozen, matching the paper's "freeze the foundation model, update
# only a small adapter" design (Sec. 3.3).
LORA_TARGET_MODULES = ['fc1', 'fc2', 'out_proj']


def _apply_lora_to_model(model, lora_rank: int):
    """
    Inject rank-`lora_rank` LoRA adapters into the frozen backbone's Linear layers.

    Journal-grade fine-tuning MUST be LoRA on a frozen foundation model (paper
    Sec. 3.3), never a full fine-tune. If peft is missing, the config matches no
    modules, or too many parameters end up trainable, this RAISES rather than
    silently full-fine-tuning (which would overwrite the pretrained weights and
    contradict the method).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError(
            "Tartan fine-tune requires `peft` for LoRA (frozen foundation model). "
            "Install it:  pip install 'peft>=0.6.0'. Refusing to full-fine-tune."
        ) from e

    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=LORA_TARGET_MODULES, bias='none'))

    n_lora  = sum(1 for n, _ in model.named_parameters() if 'lora_' in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if n_lora == 0 or n_train == 0:
        raise RuntimeError(
            f"Tartan fine-tune: LoRA matched no modules "
            f"(target_modules={LORA_TARGET_MODULES}). Refusing to full-fine-tune.")
    if n_train / n_total > 0.10:
        raise RuntimeError(
            f"Tartan fine-tune: {n_train:,}/{n_total:,} trainable "
            f"({100*n_train/n_total:.1f}%) — backbone not frozen; refusing to proceed.")
    model.print_trainable_parameters()
    print(f"Tartan fine-tune: LoRA r={lora_rank} on {LORA_TARGET_MODULES} "
          f"({n_train:,}/{n_total:,} trainable); backbone + heads frozen.")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def _set_seed(seed):
    """Seed Python/NumPy/Torch RNGs for reproducible training. No-op if seed is None."""
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"  RNG seed = {seed} (reproducible training)")


def train(args):
    _set_seed(getattr(args, 'seed', 42))
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
    val_metric_path = output_dir / out_name.replace('.pt', '_val_metric.pt')

    best_val = float('inf')

    # Journal-metric validation hook (display-only — not used in backprop).
    _val_hook = None
    if val_seq is not None and args.val_metric_every > 0:
        try:
            import os as _os
            import importlib
            from dl_filters._validation import (validate_with_journal_metric,
                                                format_val_line)
            _tartan_runner = importlib.import_module(
                'dl_filters.tartan_imu.tartan_runner')

            def _val_hook(_epoch, _model):
                _model.eval()
                _save_lora(_model, _epoch, None, val_metric_path)
                _prev = _os.environ.get('TARTAN_IMU_LORA')
                _os.environ['TARTAN_IMU_LORA'] = str(val_metric_path)
                try:
                    _m = validate_with_journal_metric(
                        filter_module=_tartan_runner, val_seq=val_seq)
                    print(format_val_line(_epoch, _m))
                except Exception as _e:
                    print(f"  [val] journal-metric hook failed: {_e}")
                finally:
                    if _prev is None:
                        _os.environ.pop('TARTAN_IMU_LORA', None)
                    else:
                        _os.environ['TARTAN_IMU_LORA'] = _prev
                _model.train()
        except Exception as _e:
            print(f"  [val] hook unavailable ({_e}) — skipping journal metric")
            _val_hook = None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        for imu_b, v_gt_b in train_loader:
            imu_b  = imu_b.to(device)
            v_gt_b = v_gt_b.to(device)
            optimizer.zero_grad()
            v_pred, log_std = model(imu_b, robot_type='car', return_sequence=True)
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
                    v_pred, log_std = model(imu_b, robot_type='car', return_sequence=True)
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

        # Journal-metric validation (display-only; never enters backprop).
        if _val_hook is not None and (
                (epoch + 1) % args.val_metric_every == 0
                or epoch == args.epochs - 1):
            _val_hook(epoch, model)

    if val_loader is None:
        _save_lora(model, args.epochs - 1, None, output_dir / out_name)

    print("Fine-tuning complete.")


def _save_lora(model, epoch, val_loss, path: Path):
    """
    Save a deployable checkpoint. LoRA adapters are merged into the (frozen)
    backbone and the resulting plain _TartanIMUModel state dict is saved, so the
    runner — which loads an unwrapped model with strict=False — applies the
    adaptation without needing peft at inference. (Saving raw lora_ tensors would
    be ignored by the runner, silently running the un-adapted base model.)
    """
    import copy
    try:
        merged = copy.deepcopy(model).merge_and_unload()   # fold BA into W, unwrap peft
        state  = merged.state_dict()
    except AttributeError:
        # Not a peft model (should not happen — LoRA is mandatory). Save as-is.
        state = model.state_dict()
    torch.save({
        'epoch':           epoch,
        'lora_state_dict': state,
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
    parser.add_argument('--val-metric-every', type=int, default=10,
                        help="Every K epochs (and at final epoch), run a "
                             "display-only validation on the held-out sequence "
                             "using the journal three-component metric "
                             "J = ATE_outage + t_rel + r_rel (default 10; "
                             "0 disables). Original NLL training loss is "
                             "unchanged.")
    parser.add_argument('--seed', type=int, default=42,
                        help="RNG seed for reproducible training (default: 42, "
                             "matching the genetic optimiser).")
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
