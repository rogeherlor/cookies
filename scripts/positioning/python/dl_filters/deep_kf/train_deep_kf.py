# -*- coding: utf-8 -*-
"""
Deep KF Training Script — Leave-One-Out CV on KITTI clean sequences.

Usage
-----
# Single LOO fold:
python train_deep_kf.py --mode loo --val-seq 01 --epochs 150 --output artifacts/deep_kf/

# Train on ALL clean sequences:
python train_deep_kf.py --mode all --epochs 150 --output artifacts/deep_kf/

Training objective (Hosseinyalamdary 2018, Eqs. 27-28)
------------------------------------------------------
Two-phase training faithful to the original DKF paper:

Phase 1: Generate ESKF posterior trajectories x_t^+ for each training sequence
          by running eskf_enhanced with GPS updates.

Phase 2: Teacher-forced sequence training:
    - Input:  x_{t-1}^+ (15D posterior state from ESKF)
    - Target: x_t^+     (15D posterior state at next timestep)
    - Loss:   weighted MSE on all 15 state components
    - The LSTM learns to predict state transitions, including the effect of
      GPS measurement updates on the state.

The model uses residual prediction (Eq. 21, μ_t = x_{t-1}^+):
    x_t^{+-} = decoder(LSTM(x_{t-1}^+)) + x_{t-1}^+

LOO Clean Sequences
-------------------
01, 04, 06, 07, 08, 09, 10

References
----------
Hosseinyalamdary, "Deep Kalman Filter: Simultaneous Multi-Sensor Integration
and Modelling; A GNSS/IMU Case Study", MDPI Sensors 2018
https://doi.org/10.3390/s18051316
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_SCRIPTS   = _REPO_ROOT / 'scripts/positioning/python'
_FILTERS   = _SCRIPTS / 'filters'

for p in [str(_HERE), str(_SCRIPTS), str(_FILTERS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl
from model import DeepKFNet

CLEAN_SEQS = ['01', '04', '06', '07', '08', '09', '10']


# ── Phase 1: Generate ESKF posterior trajectories ─────────────────────────────

def generate_eskf_posteriors(nav):
    """
    Run ESKF enhanced on a sequence to produce the posterior state trajectory.

    Returns
    -------
    x_post : (N, 15) float32 — [p(3), v(3), rpy(3), b_acc(3), b_gyr(3)]
    """
    import eskf_enhanced

    result = eskf_enhanced.run(nav)
    x_post = np.concatenate([
        result['p'],         # (N, 3) position ENU
        result['v'],         # (N, 3) velocity ENU
        result['r'],         # (N, 3) roll, pitch, yaw
        result['bias_acc'],  # (N, 3) accel bias FLU
        result['bias_gyr'],  # (N, 3) gyro bias FLU
    ], axis=1)  # (N, 15)

    # Unwrap yaw to avoid 2π discontinuities in training targets
    x_post[:, 8] = np.unwrap(x_post[:, 8])  # yaw is index 8 (rpy[2])

    return x_post.astype(np.float32)


# ── Phase 2: Teacher-forced sequence training ─────────────────────────────────

def compute_component_weights(x_post_list):
    """
    Compute per-component inverse-variance weights for balanced MSE loss.

    The 15D state has very different scales (position ~100m, biases ~0.01).
    Weighting inversely by variance balances gradient contributions.

    Parameters
    ----------
    x_post_list : list of (N_i, 15) arrays

    Returns
    -------
    weights : (15,) tensor — normalised so mean(weights) = 1
    """
    all_data = np.concatenate(x_post_list, axis=0)  # (N_total, 15)
    # Compute variance of state increments (what the residual network predicts)
    deltas = np.diff(all_data, axis=0)  # (N_total-1, 15)
    var = np.var(deltas, axis=0) + 1e-12  # avoid division by zero
    inv_var = 1.0 / var
    inv_var = inv_var / inv_var.mean()  # normalise so mean weight = 1
    return torch.from_numpy(inv_var.astype(np.float32))


def run_sequence_teacher_forced(x_post, model, optimizer, device,
                                tbptt_len, weights, training=True):
    """
    Run one epoch pass on a single sequence with teacher-forced LSTM training.

    At each step t:
        input  = x_post[t-1]  (15D posterior state)
        target = x_post[t]    (15D next posterior state)
        pred   = model(input) (15D predicted state, with residual)
        loss   = weighted_MSE(pred, target)

    Parameters
    ----------
    x_post   : (N, 15) float32 — ESKF posterior trajectory
    model    : DeepKFNet
    optimizer: torch optimizer
    device   : 'cpu' or 'cuda'
    tbptt_len: int — TBPTT segment length (number of steps)
    weights  : (15,) tensor — per-component MSE weights
    training : bool — if True, backward() + step()

    Returns
    -------
    total_loss : float — accumulated loss over the sequence
    """
    N = x_post.shape[0]
    x_t = torch.from_numpy(x_post).to(device)  # (N, 15)
    w = weights.to(device)  # (15,)

    hidden = model.init_hidden(batch_size=1, device=device)
    hidden = tuple(h.detach() for h in hidden)

    total_loss  = 0.
    step_count  = 0
    loss_seg    = None

    for t in range(1, N):
        # Teacher forcing: always feed the ground-truth posterior
        input_t  = x_t[t - 1].unsqueeze(0)   # (1, 15)
        target_t = x_t[t].unsqueeze(0)        # (1, 15)

        pred_t, hidden = model(input_t, hidden)  # (1, 15)

        # Weighted MSE loss
        step_loss = ((pred_t - target_t) ** 2 * w).mean()

        loss_seg = step_loss if loss_seg is None else loss_seg + step_loss
        step_count += 1

        # TBPTT boundary
        if step_count >= tbptt_len and loss_seg is not None:
            if training:
                optimizer.zero_grad()
                loss_seg.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
                hidden = tuple(h.detach() for h in hidden)
            total_loss += loss_seg.item()
            loss_seg    = None
            step_count  = 0

    # Final partial segment
    if loss_seg is not None:
        if training:
            optimizer.zero_grad()
            loss_seg.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
        total_loss += loss_seg.item()

    return total_loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    if args.mode == 'loo':
        if args.val_seq not in CLEAN_SEQS:
            raise ValueError(f"--val-seq must be one of {CLEAN_SEQS}")
        train_seqs = [s for s in CLEAN_SEQS if s != args.val_seq]
        val_seqs   = [args.val_seq]
        out_name   = f'fold_{args.val_seq}.pt'
    else:
        train_seqs = CLEAN_SEQS
        val_seqs   = []
        out_name   = 'deep_kf.pt'

    print(f"Mode={args.mode}  train={train_seqs}  val={val_seqs}")

    # ── Load nav data ─────────────────────────────────────────────────────
    print("Loading sequences ...")
    train_navs = []
    for seq in train_seqs:
        try:
            train_navs.append(dl.get_kitti_dataset(seq, sample_rate=100.0))
            print(f"  Loaded seq {seq}")
        except Exception as e:
            print(f"  WARNING: failed to load seq {seq}: {e}")

    val_navs = []
    for seq in val_seqs:
        try:
            val_navs.append(dl.get_kitti_dataset(seq, sample_rate=100.0))
        except Exception as e:
            print(f"  WARNING: failed to load val seq {seq}: {e}")

    # ── Phase 1: Generate ESKF posteriors ─────────────────────────────────
    print("Phase 1: Generating ESKF posterior trajectories ...")
    train_posteriors = []
    for i, nav in enumerate(train_navs):
        x_post = generate_eskf_posteriors(nav)
        train_posteriors.append(x_post)
        print(f"  Seq {train_seqs[i]}: {x_post.shape[0]} samples, "
              f"pos range [{x_post[:,:3].min():.1f}, {x_post[:,:3].max():.1f}] m")

    val_posteriors = []
    for i, nav in enumerate(val_navs):
        x_post = generate_eskf_posteriors(nav)
        val_posteriors.append(x_post)

    # Compute per-component weights from training data
    weights = compute_component_weights(train_posteriors)
    print(f"  Component weights: {weights.numpy().round(3)}")

    # ── Phase 2: Train LSTM ───────────────────────────────────────────────
    model = DeepKFNet(nav_state_dim=15,
                      hidden_dim=args.latent_dim, num_layers=2)
    model = model.to(device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')

    print(f"\nPhase 2: Training LSTM (teacher-forced, {args.epochs} epochs) ...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.
        for x_post in train_posteriors:
            train_loss += run_sequence_teacher_forced(
                x_post, model, optimizer, device,
                tbptt_len=args.tbptt_len, weights=weights,
                training=True,
            )
        scheduler.step()

        if val_posteriors:
            model.eval()
            val_loss = 0.
            with torch.no_grad():
                for x_post in val_posteriors:
                    val_loss += run_sequence_teacher_forced(
                        x_post, model, optimizer, device,
                        tbptt_len=args.tbptt_len, weights=weights,
                        training=False,
                    )
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': {
                        'latent_dim': args.latent_dim,
                        'num_layers': 2,
                        'nav_state_dim': 15,
                    },
                }, output_dir / out_name)
                print(f"  -> saved best ({out_name})")
        else:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if not val_posteriors:
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'config': {
                'latent_dim': args.latent_dim,
                'num_layers': 2,
                'nav_state_dim': 15,
            },
        }, output_dir / out_name)

    print("Training complete.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Deep KF LSTM training on KITTI")
    parser.add_argument('--mode',       choices=['loo', 'all'], default='loo')
    parser.add_argument('--val-seq',    default='01')
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int,   default=128)
    parser.add_argument('--tbptt-len',  type=int,   default=200,
                        help="TBPTT segment length (number of IMU steps)")
    parser.add_argument('--output',     default=str(_REPO_ROOT / 'artifacts/deep_kf'))
    parser.add_argument('--resume',     default=None)
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
