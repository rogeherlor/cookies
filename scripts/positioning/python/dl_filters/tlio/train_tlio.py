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

Loss (identical to external/tlio network/losses.get_loss)
----
Single diagonal-Gaussian NLL throughout, with a phase switch:
      L = (dp_gt - dp_pred)² / (2·exp(2·logstd)) + logstd          (per-axis)
  * epoch <  phase_switch : log-std is DETACHED → only the mean is supervised
                            (equivalent to MSE up to the fixed-σ weighting).
  * epoch >= phase_switch : mean and log-std train jointly.
  logstd is floored at MIN_LOG_STD = log(1e-3).  Original hard-codes
  phase_switch = 10 (--mse-epochs).

Both dp_gt and dp_pred are in the gravity-aligned frame.

Augmentation (per batch, original transform order)
-------------------------------------------------
  1. bias shift   — ±0.05 rad/s gyro, ±0.2 m/s² accel   (TransformAddNoiseBias)
  2. gravity tilt — random ≤5° about a random horizontal axis (TransformPerturbGravity)

Deliberate deviations from external/tlio (kept — justified by the vehicle domain)
--------------------------------------------------------------------------------
  * Batch size 256 (vs original 1024): KITTI's clean vehicle set is far smaller
    than TLIO's pedestrian corpus, so a smaller batch gives more updates/epoch.
  * Gravity removed from the accel channel (build_windows subtracts 9.81; the
    original keeps full specific force): car motion accel is small relative to
    gravity, so removing it emphasises the motion signal.  _perturb_gravity
    re-adds g before tilting so the augmentation stays faithful.
  * Yaw-normalised (heading-relative) input frame instead of the original's
    world frame + random-yaw augmentation (TransformInYawPlane): R_ga removes
    only roll/pitch, leaving a heading-relative frame, which already gives the
    planar-rotation invariance the yaw augmentation provides — so that transform
    is intentionally omitted as redundant.
Everything else (Adam lr, no weight decay, ReduceLROnPlateau, grad-clip 0.1,
loss schedule, ResNet1D, 1 s @ 200 Hz window) matches the original repo.

LOO Clean Sequences
-------------------
01, 04, 06, 07, 08, 09, 10  (sequences 00, 02, 05 have data gaps; 03 no raw data)

References
----------
Liu et al., "TLIO: Tight Learned Inertial Odometry", IEEE RA-L 2020
https://doi.org/10.1109/LRA.2020.3007421
Original code: https://github.com/CathIAS/TLIO (cloned to external/tlio/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

_GRAVITY = 9.81   # m/s² — must match the gravity removed in tlio_dataset.build_windows


def _augment_imu(imu_w, accel_range=0.2, gyro_range=0.05):
    """
    Add per-batch random bias offsets to simulate IMU sensor error.
    Port of external/tlio TransformAddNoiseBias (accel_bias_range=0.2,
    gyro_bias_range=0.05); channel order [gyro(0:3) | accel(3:6)] matches the
    original.  Offsets are uniform in ±range and constant over the window.
    """
    B = imu_w.shape[0]
    imu_w = imu_w.clone()
    imu_w[:, 0:3, :] += (torch.rand(B, 3, 1, device=imu_w.device) * 2 - 1) * gyro_range
    imu_w[:, 3:6, :] += (torch.rand(B, 3, 1, device=imu_w.device) * 2 - 1) * accel_range
    return imu_w


def _so3_exp(rvec):
    """Batched SO(3) exponential map (Rodrigues).  rvec (B,3) → R (B,3,3)."""
    theta      = torch.linalg.norm(rvec, dim=1, keepdim=True)        # (B,1)
    theta_safe = theta.clamp_min(1e-8)
    k          = rvec / theta_safe                                   # (B,3) unit axis
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    O          = torch.zeros_like(kx)
    K = torch.stack([O, -kz, ky, kz, O, -kx, -ky, kx, O], dim=1).reshape(-1, 3, 3)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand(rvec.shape[0], 3, 3)
    s = torch.sin(theta).unsqueeze(-1)                               # (B,1,1)
    c = (1.0 - torch.cos(theta)).unsqueeze(-1)
    return I + s * K + c * (K @ K)


def _perturb_gravity(imu_w, theta_range_deg=5.0):
    """
    Port of external/tlio TransformPerturbGravity (perturb_gravity_theta_range
    default 5.0): rotate the whole IMU window by a random tilt of magnitude
    U[0, theta_range_deg] about a random horizontal axis, simulating error in
    the estimated gravity direction.  This hardens the network against the
    roll/pitch (attitude) error it will see online, where gravity alignment uses
    the *estimated* attitude rather than ground truth.

    NOTE on gravity: the original rotates the gravity-*containing* specific force,
    so a 5° tilt injects ~g·sin(5°) ≈ 0.85 m/s² of horizontal accel — the
    meaningful signal.  Our windows have gravity removed (build_windows subtracts
    9.81 from accel_z), so we re-add g, rotate, then subtract g again to reproduce
    the original's effect on the motion accel.
    """
    B   = imu_w.shape[0]
    dev = imu_w.device
    dt  = imu_w.dtype
    azim  = torch.rand(B, device=dev, dtype=dt) * 2.0 * np.pi
    theta = torch.rand(B, device=dev, dtype=dt) * (np.pi * theta_range_deg / 180.0)
    axis  = torch.stack([torch.cos(azim), torch.sin(azim),
                         torch.zeros_like(azim)], dim=1)             # (B,3) horizontal
    R = _so3_exp(theta.unsqueeze(1) * axis)                          # (B,3,3)

    out   = imu_w.clone()
    accel = out[:, 3:6, :].clone()
    accel[:, 2, :] += _GRAVITY                                       # re-add gravity reaction
    out[:, 0:3, :] = torch.einsum('bij,bjt->bit', R, out[:, 0:3, :])  # rotate gyro
    accel = torch.einsum('bij,bjt->bit', R, accel)                  # rotate full accel
    accel[:, 2, :] -= _GRAVITY                                       # back to motion-only accel
    out[:, 3:6, :] = accel
    return out


MIN_LOG_STD = float(np.log(1e-3))   # logstd floor — external/tlio network/losses.py


def loss_distribution_diag(dp_pred, logstd, dp_gt):
    """
    Diagonal-Gaussian NLL, identical to external/tlio loss_distribution_diag:
        L = (dp_gt - dp_pred)² / (2·exp(2·logstd)) + logstd        (per-axis, B×3)
    logstd floored at MIN_LOG_STD = log(1e-3) as in the original.
    """
    logstd = torch.clamp(logstd, min=MIN_LOG_STD)
    return (dp_pred - dp_gt) ** 2 / (2. * torch.exp(2. * logstd)) + logstd


def get_loss(dp_pred, logstd, dp_gt, epoch, phase_switch):
    """
    Original TLIO schedule (external/tlio network/losses.get_loss): before
    `phase_switch` the log-std is detached so only the mean is supervised
    (≈ MSE up to the fixed-σ weighting); afterwards mean and log-std train
    jointly.  The original hard-codes phase_switch = 10.
    """
    if epoch < phase_switch:
        logstd = logstd.detach()
    return loss_distribution_diag(dp_pred, logstd, dp_gt).mean()


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

    # Determine sequences.
    # NESTED split: --val-seq names the LOO HELD-OUT (test) sequence, which is
    # never loaded here. The inner validation sequence that drives best-checkpoint
    # selection and ReduceLROnPlateau is carved out of the remaining training
    # sequences by dl_filters._validation.inner_split — see that function for why
    # validating on the held-out sequence biased these rows.
    if args.mode == 'loo':
        if args.val_seq not in CLEAN_SEQS_ACTIVE:
            raise ValueError(f"--val-seq must be one of {CLEAN_SEQS_ACTIVE}")
        from dl_filters._validation import inner_split
        held_out_seq = args.val_seq
        train_seqs, val_seq = inner_split(CLEAN_SEQS_ACTIVE, held_out_seq)
        out_name   = f'fold_{args.val_seq}.pt'
    else:  # all
        held_out_seq = None
        train_seqs = CLEAN_SEQS_ACTIVE
        val_seq    = None
        out_name   = 'tlio_resnet.pt'

    print(f"Dataset={args.dataset}  Mode={args.mode}  train={train_seqs}  "
          f"inner_val={val_seq}  held_out(test, unused)={held_out_seq}")

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

    # Optimizer + scheduler match external/tlio network/train.py exactly:
    # plain Adam (no weight decay) + ReduceLROnPlateau(factor=0.1, patience=10).
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-12)

    phase_switch = args.mse_epochs    # log-std detached (mean-only) before this epoch

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / out_name.replace('.pt', '_ckpt.pt')
    val_metric_path = output_dir / out_name.replace('.pt', '_val_metric.pt')

    best_val_loss = float('inf')
    # SELECTION criterion. best_J is the three-component journal cost
    #     J = ATE_outage/1m + t_rel/1% + r_rel/1(deg/km)
    # measured by closing the loop on the INNER-VALIDATION sequence with the
    # standard 40s/60s outage — i.e. the same objective ins_genetic_cv.py
    # minimises for the seven classical filters.
    #
    # It replaces one-step NLL validation loss for choosing which epoch to
    # deploy. NLL cannot see this at all: these models are trained OUTAGE-FREE
    # (run_dl_training.sh: "outages are simulated only at evaluation time"), so
    # a one-step density score on clean data carries no information about the
    # dead-reckoning behaviour the tables actually report. Selecting on it was
    # effectively arbitrary with respect to the reported metric, and measurably
    # so — Deep KF's LOO outage mean moved between 86 m and 557 m depending only
    # on which epoch that blind criterion happened to pick.
    best_J = float('inf')
    best_J_epoch = None

    # Journal-metric hook. NOT used in backprop — it selects the checkpoint.
    # It scores the INNER validation sequence, never the LOO held-out one.
    _val_hook = None
    if args.mode == 'loo' and val_seq is not None and args.val_metric_every > 0:
        try:
            import os as _os
            import importlib
            from dl_filters._validation import (validate_with_journal_metric,
                                                format_val_line)
            _tlio_runner = importlib.import_module(
                'dl_filters.tlio.tlio_runner')

            def _val_hook(_epoch, _model):
                """Score this epoch on the inner-val sequence; returns J (inf on
                failure). The weights have to reach the runner through a file
                because the runner resolves its own checkpoint — writing them to
                val_metric_path and pointing TLIO_WEIGHTS at it is the same
                mechanism the evaluation uses, so the number measured here is
                produced by the identical code path that produces the table."""
                _model.eval()
                torch.save({'epoch': _epoch,
                            'model_state_dict': _model.state_dict(),
                            'args': vars(args)},
                           val_metric_path)
                _prev = _os.environ.get('TLIO_WEIGHTS')
                _os.environ['TLIO_WEIGHTS'] = str(val_metric_path)
                _J = float('inf')
                try:
                    _m = validate_with_journal_metric(
                        filter_module=_tlio_runner, val_seq=val_seq)
                    print(format_val_line(_epoch, _m))
                    _J = float(_m.get('J', float('inf')))
                except Exception as _e:
                    print(f"  [val] journal-metric hook failed: {_e}")
                finally:
                    if _prev is None:
                        _os.environ.pop('TLIO_WEIGHTS', None)
                    else:
                        _os.environ['TLIO_WEIGHTS'] = _prev
                _model.train()
                return _J
        except Exception as _e:
            print(f"  [val] hook unavailable ({_e}) — skipping journal metric")
            _val_hook = None

    for epoch in range(start_epoch, args.epochs):
        # ── Training ───────────────────────────────────────────────────
        model.train()
        total_loss = 0.
        for imu_w, dp_target in train_loader:
            imu_w     = _augment_imu(imu_w.to(device))
            if args.perturb_gravity:
                imu_w = _perturb_gravity(imu_w, args.gravity_theta_deg)
            dp_target = dp_target.to(device)

            optimizer.zero_grad()
            mean_pred, logstd_pred = model(imu_w)

            loss = get_loss(mean_pred, logstd_pred, dp_target, epoch, phase_switch)

            loss.backward()
            # Grad clip 0.1 matches external/tlio network/train.py.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1,
                                     error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * imu_w.size(0)

        avg_train = total_loss / len(train_ds)
        phase = 'NLL(μ)' if epoch < phase_switch else 'NLL(μ,σ)'
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
                    loss = get_loss(mean_pred, logstd_pred, dp_target,
                                    epoch, phase_switch)
                    val_loss += loss.item() * imu_w.size(0)
            avg_val = val_loss / len(val_ds)
            # LR schedule still follows the inner-val NLL: it is a smooth,
            # every-epoch signal, which is what a plateau detector needs. Only
            # CHECKPOINT SELECTION moves to the journal metric, below.
            scheduler.step(avg_val)
            best_val_loss = min(best_val_loss, avg_val)
            if _should_print:
                print(f"Epoch {epoch+1:4d}/{args.epochs} [{phase}]  "
                      f"train={avg_train:.4f}  val={avg_val:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
        else:
            scheduler.step(avg_train)   # no val set (--mode all): plateau on train loss
            if _should_print:
                print(f"Epoch {epoch+1:4d}/{args.epochs} [{phase}]  "
                      f"train={avg_train:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Periodic checkpoint (every 20 epochs)
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'args':             vars(args),
            }, ckpt_path)

        # Journal-metric validation — THIS is what selects the deployed epoch.
        # Never enters backprop; it only decides which weights are kept.
        if _val_hook is not None and (
                (epoch + 1) % args.val_metric_every == 0
                or epoch == args.epochs - 1):
            _J = _val_hook(epoch, model)
            if _J < best_J:
                best_J, best_J_epoch = _J, epoch
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss':         avg_val if val_loader is not None else None,
                    'journal_J':        _J,
                    'selection':        'journal_metric_inner_val',
                    'inner_val_seq':    val_seq,
                    'args':             vars(args),
                }, output_dir / out_name)
                print(f"  → saved best weights ({out_name})  J={_J:.4f} @ epoch {epoch}")

    # Every scored epoch was rejected (J=inf everywhere, e.g. the filter
    # diverged on the inner-val sequence at every checkpoint). Saving the last
    # epoch anyway would look identical on disk to a legitimate selection, so
    # fail instead — a fold with no valid checkpoint must not silently become a
    # table row.
    if _val_hook is not None and best_J_epoch is None:
        raise RuntimeError(
            f"No checkpoint was selected for fold {held_out_seq}: the journal "
            f"metric was inf at every evaluated epoch on inner-val sequence "
            f"{val_seq}. Refusing to deploy an unselected checkpoint.")
    if _val_hook is not None:
        print(f"Selected epoch {best_J_epoch} (J={best_J:.4f} on inner-val "
              f"seq {val_seq}) -> {out_name}")

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
    parser.add_argument('--mse-epochs', type=int, default=10,
                        help='Phase switch: before this epoch the log-std is detached '
                             '(mean-only supervision), after it mean+log-std train '
                             'jointly (default: 10 — matches external/tlio '
                             'network/losses.get_loss).')
    parser.add_argument('--output',     default=str(_REPO_ROOT / 'artifacts/tlio'),
                        help="Directory to save model weights")
    parser.add_argument('--resume',     default=None,
                        help="Path to checkpoint .pt to resume training from")
    parser.add_argument('--val-metric-every', type=int, default=10,
                        help="Every K epochs (and at final epoch), run a "
                             "display-only validation on the held-out sequence "
                             "using the journal three-component metric "
                             "J = ATE_outage + t_rel + r_rel (default 10; "
                             "0 disables). Original MSE/NLL training loss is "
                             "unchanged.")
    parser.add_argument('--seed', type=int, default=42,
                        help="RNG seed for reproducible training (default: 42, "
                             "matching the genetic optimiser).")
    parser.add_argument('--perturb-gravity', dest='perturb_gravity',
                        action='store_true', default=True,
                        help="Apply random gravity-direction (tilt) augmentation "
                             "during training (default: on — matches original TLIO "
                             "perturb_gravity=True).")
    parser.add_argument('--no-perturb-gravity', dest='perturb_gravity',
                        action='store_false',
                        help="Disable the gravity-direction augmentation.")
    parser.add_argument('--gravity-theta-deg', type=float, default=5.0,
                        help="Max tilt magnitude in degrees for gravity-direction "
                             "augmentation (default: 5.0 — matches original TLIO "
                             "perturb_gravity_theta_range).")
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
