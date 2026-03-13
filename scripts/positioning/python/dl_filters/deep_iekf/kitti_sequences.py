"""
kitti_sequences.py — Adapter between NavigationData and the AI-IMU BaseDataset format.

Credit / License
----------------
The AI-IMU training infrastructure this adapter targets is the work of Martin Brossard,
Axel Barrau, and Silvère Bonnabel, MIT License (external/ai-imu-dr/LICENSE).
Original repo: https://github.com/mbrossar/ai-imu-dr
Reference: Brossard, Barrau, Bonnabel — "AI-IMU Dead-Reckoning", IEEE TIV 2020,
           DOI 10.1109/TIV.2020.2980758

The AI-IMU training infrastructure (train_torch_filter.py) expects a dataset object
whose get_data() method returns (t, ang_gt, p_gt, v_gt, u) as torch tensors, and
whose normalize_factors property contains {'u_loc', 'u_std'}.

This module provides:
  - nav_data_to_aimu_tensors(): quick conversion for single-sequence inference
  - SingleSequenceDataset: minimal dataset wrapper for one NavigationData object
  - compute_normalization(): compute IMU normalization factors from NavigationData

NOTE on 10 Hz vs 100 Hz
-----------------------
Our KITTI .mat files contain the synchronized data at 10 Hz.  The AI-IMU paper uses
raw KITTI OXTS data at 100 Hz (dt = 0.01 s).  At 10 Hz the CNN window of 15 samples
covers 1.5 s instead of 0.15 s, and the model's sense of "local" IMU dynamics changes.

For proper reproduction of paper results, train with raw 100 Hz data via train_ai_imu.py
(requires the full KITTI raw dataset).  The inference wrapper (iekf_ai_imu.py) works at
any sample rate; performance will differ from the paper if run at 10 Hz.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Path to AI-IMU source
_AI_IMU_SRC = Path(__file__).parent.parent.parent.parent.parent.parent / 'external/ai-imu-dr/src'


def _add_aimu_path():
    src = str(_AI_IMU_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def nav_data_to_aimu_tensors(nav_data):
    """
    Convert NavigationData to tensors expected by AI-IMU.

    Returns
    -------
    t       : (N,)   float64 tensor — timestamps [s]
    ang_gt  : (N,3)  float64 tensor — [roll, pitch, yaw] [rad]
    p_gt    : (N,3)  float64 tensor — ENU position [m], origin-relative
    v_gt    : (N,3)  float64 tensor — ENU velocity [m/s]
    u       : (N,6)  float64 tensor — [gyro_xyz, accel_xyz] in body (FLU) frame

    Notes
    -----
    The AI-IMU body frame is FLU and navigation frame is ENU — exactly matching ours.
    IMU input u = [gyro_flu(3), accel_flu(3)], same as in the AI-IMU OXTS extraction
    (gyro_bis = wx,wy,wz ≈ FLU axes for the RT3003 on KITTI).
    """
    import pymap3d as pm

    N = len(nav_data.accel_flu)

    # Timestamps
    if nav_data.time is not None:
        t = nav_data.time.astype(np.float64)
    else:
        dt = 1.0 / nav_data.sample_rate
        t = np.arange(N, dtype=np.float64) * dt

    # Orientation [roll, pitch, yaw]
    ang_gt = nav_data.orient.astype(np.float64)  # (N, 3)

    # ENU position (origin-relative)
    lla = nav_data.lla
    lla0 = nav_data.lla0
    e, n, u = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2],
                               lla0[0], lla0[1], lla0[2])
    p_gt = np.column_stack([e, n, u]).astype(np.float64)
    p_gt -= p_gt[0]  # origin-relative (AI-IMU convention)

    # ENU velocity
    v_gt = nav_data.vel_enu.astype(np.float64)

    # IMU: [gyro_flu, accel_flu]
    u_arr = np.hstack([nav_data.gyro_flu, nav_data.accel_flu]).astype(np.float64)

    return (
        torch.from_numpy(t),
        torch.from_numpy(ang_gt),
        torch.from_numpy(p_gt),
        torch.from_numpy(v_gt),
        torch.from_numpy(u_arr),
    )


def compute_normalization(nav_data):
    """
    Compute IMU normalization factors (mean and std) from NavigationData.

    Returns {'u_loc': tensor(6,), 'u_std': tensor(6,)}
    """
    u_arr = np.hstack([nav_data.gyro_flu, nav_data.accel_flu]).astype(np.float64)
    u_loc = torch.from_numpy(np.mean(u_arr, axis=0))
    u_std = torch.from_numpy(np.std(u_arr, axis=0))
    u_std[u_std < 1e-6] = 1.0  # avoid division by zero
    return {'u_loc': u_loc, 'u_std': u_std}


class SingleSequenceDataset:
    """
    Minimal dataset wrapper around a single NavigationData object.

    Satisfies the interface expected by AI-IMU's training infrastructure:
      dataset.get_data(name) → (t, ang_gt, p_gt, v_gt, u)  — torch tensors
      dataset.normalize_factors → {'u_loc', 'u_std'}
      dataset.add_noise(u) → u with injected Gaussian noise
      dataset.datasets_train_filter  → dict {name: [start, end]}
      dataset.datasets_validatation_filter → dict

    WARNING: Training on a single sequence will overfit.  This is only meant
    for quick sanity checks or fine-tuning.  For proper training use the full
    KITTI raw dataset via train_ai_imu.py --mode kitti.
    """

    sigma_gyro  = 1e-4   # rad/s  noise std for data augmentation
    sigma_acc   = 1e-3   # m/s²   noise std for data augmentation

    def __init__(self, nav_data, name='single_sequence', val_fraction=0.1):
        self.name = name
        self._t, self._ang_gt, self._p_gt, self._v_gt, self._u = \
            nav_data_to_aimu_tensors(nav_data)

        self.normalize_factors = compute_normalization(nav_data)

        N = self._t.shape[0]
        split = int(N * (1 - val_fraction))
        self.datasets_train_filter      = {name: [0, split]}
        self.datasets_validatation_filter = {name: [split, N]}
        self.datasets = [name]

    def get_data(self, dataset_name):
        return self._t, self._ang_gt, self._p_gt, self._v_gt, self._u

    def add_noise(self, u):
        """Add small Gaussian noise to IMU for data augmentation during training."""
        noise = torch.zeros_like(u)
        noise[:, :3] = self.sigma_gyro * torch.randn_like(u[:, :3])
        noise[:, 3:] = self.sigma_acc  * torch.randn_like(u[:, 3:])
        return u + noise

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def dump(self, data, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
