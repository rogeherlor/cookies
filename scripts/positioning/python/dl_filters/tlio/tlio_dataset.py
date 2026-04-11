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

Frequency
---------
The original TLIO paper uses 200 Hz IMU with 1 s windows (W=200).  Our KITTI
pickles are at 100 Hz.  build_windows() accepts a target_rate parameter: when
set to 200.0, IMU is linearly upsampled before windowing so that W=200 covers
exactly 1 s — matching the original paper's architecture and time scale.
"""

import numpy as np
from scipy.interpolate import interp1d as _interp1d


TARGET_HZ = 200.0   # original TLIO paper frequency


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


# ── Upsampling ───────────────────────────────────────────────────────────────

def upsample_imu(accel_flu, gyro_flu, orient, vel_enu, src_rate, target_rate=TARGET_HZ):
    """
    Linearly interpolate IMU and navigation signals from src_rate to target_rate Hz.

    Parameters
    ----------
    accel_flu  : (N, 3) accelerometer in FLU body frame
    gyro_flu   : (N, 3) gyroscope in FLU body frame
    orient     : (N, 3) [roll, pitch, yaw] in radians
    vel_enu    : (N, 3) ENU velocity [m/s]
    src_rate   : source sample rate [Hz]
    target_rate: target sample rate [Hz] (default TARGET_HZ = 200)

    Returns
    -------
    accel_up, gyro_up, orient_up, vel_up : upsampled arrays at target_rate
    t_up : (N_up,) upsampled timestamps [s]
    """
    N = accel_flu.shape[0]
    t_src = np.arange(N) / src_rate
    t_up  = np.arange(0., t_src[-1], 1.0 / target_rate)

    kw = dict(axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    accel_up  = _interp1d(t_src, accel_flu,  **kw)(t_up)
    gyro_up   = _interp1d(t_src, gyro_flu,   **kw)(t_up)
    vel_up    = _interp1d(t_src, vel_enu,    **kw)(t_up)
    orient_up = np.stack(
        [np.interp(t_up, t_src, orient[:, i]) for i in range(3)],
        axis=1,
    )

    return accel_up, gyro_up, orient_up, vel_up, t_up


# ── Window builder ───────────────────────────────────────────────────────────

def build_windows(nav_data, window_size: int, stride: int,
                  target_rate: float = None):
    """
    Build gravity-aligned IMU windows from NavigationData.

    Parameters
    ----------
    nav_data    : NavigationData — data_loader output (FLU / ENU).
    window_size : int — number of IMU samples per window (W), in target_rate Hz.
    stride      : int — step between window starts (S), in target_rate Hz.
    target_rate : float or None — if provided, upsample IMU to this rate before
                  windowing (use TARGET_HZ=200 to match the original TLIO paper).
                  window_size and stride must be in target_rate samples.
                  If None, windows are built at nav_data.sample_rate.

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
        Index into the (possibly upsampled) arrays at which each window begins.
    """
    src_rate = nav_data.sample_rate

    if target_rate is not None and abs(target_rate - src_rate) > 0.5:
        accel_flu, gyro_flu, orient, vel_enu, _ = upsample_imu(
            nav_data.accel_flu, nav_data.gyro_flu, nav_data.orient,
            nav_data.vel_enu, src_rate, target_rate,
        )
        dt = 1.0 / target_rate
    else:
        accel_flu = nav_data.accel_flu
        gyro_flu  = nav_data.gyro_flu
        orient    = nav_data.orient
        vel_enu   = nav_data.vel_enu
        dt        = 1.0 / src_rate

    N = accel_flu.shape[0]

    starts = np.arange(0, N - window_size, stride)
    M = len(starts)

    imu_windows  = np.zeros((M, 6, window_size), dtype=np.float32)
    dp_gt        = np.zeros((M, 3), dtype=np.float64)
    Rz_list      = []
    window_starts = starts.copy()

    # Ground-truth position in ENU from velocity integration
    p_gt = _integrate_velocity(vel_enu, dt)   # (N, 3)

    for k, i in enumerate(starts):
        roll_i  = orient[i, 0]
        pitch_i = orient[i, 1]
        yaw_i   = orient[i, 2]

        # Gravity-alignment rotation (removes roll and pitch tilt)
        R_ga = _Ry(pitch_i) @ _Rx(roll_i)

        # Rotate window of IMU samples to gravity-aligned frame
        accel_w = accel_flu[i:i + window_size]   # (W, 3)
        gyro_w  = gyro_flu[i:i + window_size]    # (W, 3)

        accel_ga = (R_ga @ accel_w.T)            # (3, W)
        gyro_ga  = (R_ga @ gyro_w.T)             # (3, W)

        # Remove gravity: at rest in gravity-aligned (Z-up) frame the
        # accelerometer reads +9.81 in Z (reaction to gravity).
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
