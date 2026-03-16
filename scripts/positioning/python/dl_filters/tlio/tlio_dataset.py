# -*- coding: utf-8 -*-
"""
Gravity-aligned IMU window builder for TLIO.

Converts NavigationData (FLU / ENU convention) into windowed tensors for
training and inference.  One window = W contiguous IMU samples; windows
overlap by (W - stride) samples.

Frame math
----------
R_ga  = Ry(pitch) @ Rx(roll)          # body (FLU) → gravity-aligned frame
accel_ga_motion = R_ga @ accel_flu - [0, 0, 9.81]  # remove gravity reaction
gyro_ga  = R_ga @ gyro_flu            # no subtraction (gyro is gravity-free)

The gravity-aligned frame shares its Z-axis with ENU-up; the only difference
from ENU is a yaw rotation.  So for rotating predicted displacement back:
    dp_enu = Rz(yaw_window_start) @ dp_ga
    Sigma_enu = Rz @ Sigma_ga @ Rz.T
"""

import numpy as np


# ── Rotation helpers ────────────────────────────────────────────────────────

def _Rx(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., ca, -sa], [0., sa, ca]])


def _Ry(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0., sa], [0., 1., 0.], [-sa, 0., ca]])


def _Rz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])


# ── Window builder ───────────────────────────────────────────────────────────

def build_windows(nav_data, window_size: int, stride: int):
    """
    Build gravity-aligned IMU windows from NavigationData.

    Parameters
    ----------
    nav_data    : NavigationData — data_loader output (FLU / ENU).
    window_size : int — number of IMU samples per window (W).
    stride      : int — step between window starts (S).

    Returns
    -------
    imu_windows  : np.ndarray (M, 6, W) float32
        [gyro_ga(3) | accel_ga_motion(3)] in gravity-aligned frame.
        Gravity component subtracted from accelerometer.
    dp_gt        : np.ndarray (M, 3) float64
        Ground-truth displacement in ENU [m] for each window.
    Rz_list      : list of (3, 3) np.ndarray
        Rz(yaw_at_window_start) matrices — rotate GA prediction back to ENU.
    window_starts : np.ndarray (M,) int
        Index into nav_data arrays at which each window begins.
    """
    accel_flu = nav_data.accel_flu   # (N, 3)
    gyro_flu  = nav_data.gyro_flu    # (N, 3)
    orient    = nav_data.orient      # (N, 3) [roll, pitch, yaw]
    N = accel_flu.shape[0]

    starts = np.arange(0, N - window_size, stride)
    M = len(starts)

    imu_windows  = np.zeros((M, 6, window_size), dtype=np.float32)
    dp_gt        = np.zeros((M, 3), dtype=np.float64)
    Rz_list      = []
    window_starts = starts.copy()

    # Ground-truth position in ENU (reconstructed from velocity integration or
    # directly from nav_data.vel_enu via position if available).
    # We use vel_enu × dt to compute incremental displacement — this is always
    # available without pymap3d and avoids any reference-point drift.
    dt   = 1.0 / nav_data.sample_rate
    p_gt = _integrate_velocity(nav_data.vel_enu, dt)   # (N, 3) ENU position

    for k, i in enumerate(starts):
        roll_i  = orient[i, 0]
        pitch_i = orient[i, 1]
        yaw_i   = orient[i, 2]

        # Gravity-alignment rotation (removes roll and pitch tilt)
        R_ga = _Ry(pitch_i) @ _Rx(roll_i)

        # Rotate window of IMU samples to gravity-aligned frame
        # accel_flu shape (W, 3) → transpose → (3, W); R_ga (3,3) @ (3,W) → (3, W)
        accel_w = accel_flu[i:i + window_size]   # (W, 3)
        gyro_w  = gyro_flu[i:i + window_size]    # (W, 3)

        accel_ga = (R_ga @ accel_w.T)            # (3, W)
        gyro_ga  = (R_ga @ gyro_w.T)             # (3, W)

        # Remove gravity: at rest in gravity-aligned (Z-up) frame the
        # accelerometer reads +9.81 in Z (reaction to gravity).
        # Subtracting removes this constant component.
        accel_ga[2, :] -= 9.81

        # Channel order: [gyro(3), accel(3)] — matches TLIO network input
        imu_windows[k, 0:3, :] = gyro_ga.astype(np.float32)
        imu_windows[k, 3:6, :] = accel_ga.astype(np.float32)

        # Ground-truth displacement in ENU
        j = min(i + window_size, N - 1)
        dp_gt[k] = p_gt[j] - p_gt[i]

        # Yaw rotation: GA → ENU
        Rz_list.append(_Rz(yaw_i))

    return imu_windows, dp_gt, Rz_list, window_starts


def _integrate_velocity(vel_enu: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate ENU velocity to produce ENU position (relative to first sample).
    Uses trapezoidal rule for consistency with strapdown propagation.
    """
    N = vel_enu.shape[0]
    p = np.zeros((N, 3), dtype=np.float64)
    for i in range(1, N):
        p[i] = p[i - 1] + 0.5 * dt * (vel_enu[i - 1] + vel_enu[i])
    return p


def ga_to_enu(dp_ga: np.ndarray, Rz: np.ndarray) -> np.ndarray:
    """Rotate displacement from gravity-aligned to ENU frame."""
    return Rz @ dp_ga


def enu_to_ga(dp_enu: np.ndarray, Rz: np.ndarray) -> np.ndarray:
    """Rotate displacement from ENU to gravity-aligned frame (Rz.T)."""
    return Rz.T @ dp_enu


def rotate_covariance(Sigma_ga: np.ndarray, Rz: np.ndarray) -> np.ndarray:
    """Rotate diagonal covariance from GA to ENU: Rz @ Sigma_ga @ Rz.T."""
    return Rz @ Sigma_ga @ Rz.T
