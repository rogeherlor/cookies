# -*- coding: utf-8 -*-
"""
Deep KF Training Script — Leave-One-Out CV on KITTI clean sequences.

Usage
-----
# Single LOO fold:
python train_deep_kf.py --mode loo --val-seq 01 --epochs 150 --output artifacts/deep_kf/

# Train on ALL clean sequences:
python train_deep_kf.py --mode all --epochs 150 --output artifacts/deep_kf/

Training objective
------------------
The LSTM learns to predict IMU bias corrections by minimising the position and
velocity error on GPS-available timesteps (where GPS provides a strong label).

For each timestep i where GPS is available:
    δb_acc_i, δb_gyr_i = decoder(LSTM(x_{i-T:i}, imu_{i-T:i}))
    acc_corr = accel_flu[i] - δb_acc_i
    gyr_corr = gyro_flu[i]  - δb_gyr_i
    [propagate strapdown with corrected IMU over GPS interval]
    L = MSE(p_propagated, p_gps) + λ_v * MSE(v_propagated, v_gps_derived)

This is a supervised regression problem: the GPS provides ground-truth labels
at 1 Hz; the LSTM learns to correct IMU biases to match them.

Training is done via TBPTT (Truncated Back-Propagation Through Time) over
segments of length tbptt_len steps.

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

for p in [str(_HERE), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl
from model import DeepKFNet

CLEAN_SEQS = ['01', '04', '06', '07', '08', '09', '10']

GRAVITY_ENU = np.array([0., 0., -9.81])


# ── Quaternion utilities ──────────────────────────────────────────────────────

def _skew(v):
    return np.array([[ 0.,-v[2], v[1]],[ v[2], 0.,-v[0]],[-v[1], v[0], 0.]])

def _qnorm(q):
    n = np.linalg.norm(q); return q/n if n > 0. else np.array([1.,0.,0.,0.])

def _qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def _qfrom_axis_angle(dtheta):
    a = np.linalg.norm(dtheta)
    if a < 1e-12: return _qnorm(np.array([1., .5*dtheta[0], .5*dtheta[1], .5*dtheta[2]]))
    ax = dtheta/a; s = np.sin(.5*a)
    return np.array([np.cos(.5*a), ax[0]*s, ax[1]*s, ax[2]*s])

def _qfrom_euler(roll, pitch, yaw):
    cr,sr = np.cos(roll/2), np.sin(roll/2)
    cp,sp = np.cos(pitch/2), np.sin(pitch/2)
    cy,sy = np.cos(yaw/2), np.sin(yaw/2)
    return _qnorm(np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))

def _qto_Rbn(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def _qto_rpy(q):
    w,x,y,z=q
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1.,1.)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])


# ── Propagation helper ────────────────────────────────────────────────────────

def propagate_step(q, pIMU, vIMU, acc_b, omega_b, Ts, use_3d=True):
    """Single IMU propagation step. Returns (q_new, p_new, v_new, Rbn)."""
    dtheta = omega_b * Ts if use_3d else np.array([0.,0.,omega_b[2]*Ts])
    q_new  = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
    Rbn    = _qto_Rbn(q_new)
    accENU = Rbn @ acc_b
    p_new  = pIMU + Ts*vIMU + 0.5*Ts**2*(accENU + GRAVITY_ENU)
    v_new  = vIMU + Ts*(accENU + GRAVITY_ENU)
    return q_new, p_new, v_new, Rbn


# ── Training ──────────────────────────────────────────────────────────────────

def run_sequence_tbptt(nav, model, optimizer, device, tbptt_len,
                       lambda_vel, gps_interval=100, use_3d=True,
                       training=True):
    """
    Run one epoch pass on a single sequence with TBPTT.

    Position and velocity are kept as differentiable torch tensors so that
    gradients flow: LSTM → db_acc_t → acc_corr_t → pIMU_t → loss_pos.
    Orientation (quaternion) is updated in numpy since we do not backprop
    through rotation.  At each GPS update the strapdown is hard-reset to the
    GPS position/velocity (detached), which caps the graph depth at ~100 IMU
    steps between GPS fixes.

    Parameters
    ----------
    training : bool
        If True, call backward() and optimizer.step().
        If False (validation), only accumulate loss values.

    Returns cumulative loss value (float).
    """
    import pymap3d as pm

    accel_flu = nav.accel_flu
    gyro_flu  = nav.gyro_flu
    orient    = nav.orient
    vel_enu   = nav.vel_enu
    lla       = nav.lla
    lla0      = nav.lla0
    N  = accel_flu.shape[0]
    Ts = 1.0 / nav.sample_rate
    T_win = int(1.0 * nav.sample_rate)   # 1-second rolling buffer for LSTM

    e, n, u = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2],
                               lla0[0], lla0[1], lla0[2])
    p_gps = np.column_stack([e, n, u])

    GRAVITY_T   = torch.tensor([0., 0., -9.81], device=device)
    accel_flu_t = torch.from_numpy(accel_flu).float().to(device)  # (N, 3)

    # Nominal state as torch tensors (grad flows through these)
    pIMU_t = torch.tensor(p_gps[0],   dtype=torch.float32, device=device)
    vIMU_t = torch.tensor(vel_enu[0], dtype=torch.float32, device=device)
    q      = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])

    hidden = model.init_hidden(batch_size=1, device=device)
    hidden = tuple(h.detach() for h in hidden)

    imu_buf_acc = np.zeros((T_win, 3))
    imu_buf_gyr = np.zeros((T_win, 3))

    total_loss  = 0.
    tbptt_count = 0
    loss_seg    = None          # None until we accumulate a GPS step with grad

    for i in range(N - 1):
        # ── LSTM modelling step (produces differentiable bias corrections) ──
        rpy_now = _qto_rpy(q)
        x_post_np = np.concatenate([
            pIMU_t.detach().cpu().numpy(),
            vIMU_t.detach().cpu().numpy(),
            rpy_now, np.zeros(3), np.zeros(3),
        ])
        imu_mean_np = np.concatenate([imu_buf_acc.mean(0), imu_buf_gyr.mean(0)])

        x_post_t   = torch.from_numpy(x_post_np).float().unsqueeze(0).to(device)
        imu_mean_t = torch.from_numpy(imu_mean_np).float().unsqueeze(0).to(device)

        bias_t, hidden = model(x_post_t, imu_mean_t, hidden)
        db_acc_t = bias_t[0, 0:3]   # (3,) — has grad_fn via LSTM
        db_gyr_t = bias_t[0, 3:6]   # (3,)

        # Gyro correction in numpy (orientation, no backprop needed)
        db_gyr_np = db_gyr_t.detach().cpu().numpy()
        gyr_corr  = gyro_flu[i] - db_gyr_np

        # Update rolling IMU buffer
        buf_idx = i % T_win
        imu_buf_acc[buf_idx] = accel_flu[i]
        imu_buf_gyr[buf_idx] = gyro_flu[i]

        # Quaternion / orientation update (numpy)
        dtheta = (gyr_corr * Ts if use_3d
                  else np.array([0., 0., gyr_corr[2] * Ts]))
        q     = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn_t = torch.from_numpy(_qto_Rbn(q)).float().to(device)   # (3, 3)

        # ── Strapdown integration in torch — grad flows from db_acc_t ──────
        acc_corr_t = accel_flu_t[i] - db_acc_t          # (3,) — keeps grad_fn
        acc_enu_t  = Rbn_t @ acc_corr_t + GRAVITY_T     # (3,)
        pIMU_t = pIMU_t + Ts * vIMU_t + 0.5 * Ts ** 2 * acc_enu_t
        vIMU_t = vIMU_t + Ts * acc_enu_t

        # ── GPS supervision ─────────────────────────────────────────────────
        if nav.gps_available[i + 1]:
            p_gps_t = torch.tensor(p_gps[i + 1],
                                   dtype=torch.float32, device=device)
            v_gps_t = torch.tensor(vel_enu[i + 1],
                                   dtype=torch.float32, device=device)

            loss_pos  = ((pIMU_t - p_gps_t) ** 2).mean()
            loss_vel  = ((vIMU_t - v_gps_t) ** 2).mean()
            step_loss = loss_pos + lambda_vel * loss_vel

            loss_seg = step_loss if loss_seg is None else loss_seg + step_loss
            tbptt_count += 1

            # Hard GPS reset: detach so the graph does not grow unboundedly.
            # Gradient of this step's loss has already been attached to loss_seg.
            pIMU_t = p_gps_t.detach().clone()
            vIMU_t = v_gps_t.detach().clone()

        # ── TBPTT boundary ──────────────────────────────────────────────────
        if tbptt_count >= tbptt_len and loss_seg is not None:
            if training:
                optimizer.zero_grad()
                loss_seg.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
                hidden = tuple(h.detach() for h in hidden)
            total_loss  += loss_seg.item()
            loss_seg     = None
            tbptt_count  = 0

    # ── Final partial segment ────────────────────────────────────────────────
    if loss_seg is not None:
        if training:
            optimizer.zero_grad()
            loss_seg.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
        total_loss += loss_seg.item()

    return total_loss


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

    # Load nav data
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

    # Model
    model = DeepKFNet(nav_state_dim=15, imu_dim=6,
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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.
        for nav in train_navs:
            train_loss += run_sequence_tbptt(
                nav, model, optimizer, device,
                tbptt_len=args.tbptt_len, lambda_vel=args.lambda_vel,
                training=True,
            )
        scheduler.step()

        if val_navs:
            model.eval()
            val_loss = 0.
            for nav in val_navs:
                val_loss += run_sequence_tbptt(
                    nav, model, optimizer, device,
                    tbptt_len=args.tbptt_len, lambda_vel=args.lambda_vel,
                    training=False,
                )
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'val_loss': val_loss}, output_dir / out_name)
                print(f"  → saved best ({out_name})")
        else:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if not val_navs:
        torch.save({'epoch': args.epochs-1,
                    'model_state_dict': model.state_dict()}, output_dir / out_name)

    print("Training complete.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Deep KF LSTM training on KITTI")
    parser.add_argument('--mode',       choices=['loo','all'], default='loo')
    parser.add_argument('--val-seq',    default='01')
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int,   default=128)
    parser.add_argument('--tbptt-len',  type=int,   default=20,
                        help="TBPTT segment length (number of GPS updates)")
    parser.add_argument('--lambda-vel', type=float, default=0.5,
                        help="Weight for velocity loss term")
    parser.add_argument('--output',     default=str(_REPO_ROOT / 'artifacts/deep_kf'))
    parser.add_argument('--resume',     default=None)
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
