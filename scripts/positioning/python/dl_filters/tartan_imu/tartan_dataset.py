# -*- coding: utf-8 -*-
"""
Tartan IMU dataset preprocessing.

Converts NavigationData (FLU / ENU) into 1-second LSTM steps at 200 Hz.
The Tartan IMU model expects:
  - Input rate: 200 Hz (we upsample from 100 Hz via linear interpolation)
  - Input per step: (N_step, 6) = (200, 6) at 200 Hz — [accel_body | gyro_body] in FLU
  - Gravity-free accel: a_input = accel_flu - R_bn @ [0, 0, -9.81]
    (Eq.1 of the paper: subtract specific force of gravity in body frame)
  - Each LSTM step covers 1 second; 10 steps = 10s total context window
  - Velocity output ground truth: (p_gt[t+1] - p_gt[t]) / dt rotated to body frame

Frame conventions
-----------------
Tartan IMU uses the same FLU body frame as this project — no frame rotation needed.
Only gravity removal (in body frame) is applied.
"""

import numpy as np
from scipy.interpolate import interp1d


TARGET_HZ     = 200      # model input rate
LSTM_STEPS    = 10       # 10 LSTM steps of 1 second each = 10s context
STEP_SECONDS  = 1.0      # seconds per LSTM step
STEP_SAMPLES  = int(TARGET_HZ * STEP_SECONDS)   # = 200 samples per step


def upsample_imu(accel: np.ndarray, gyro: np.ndarray,
                 src_rate: float, tgt_rate: float = TARGET_HZ) -> tuple:
    """
    Upsample 1-D IMU arrays from src_rate to tgt_rate using linear interpolation.

    Parameters
    ----------
    accel, gyro : (N, 3) arrays at src_rate
    src_rate    : float — source sample rate [Hz]
    tgt_rate    : float — target sample rate [Hz] (default 200)

    Returns
    -------
    accel_up, gyro_up : (N_up, 3) float64 arrays at tgt_rate
    t_up              : (N_up,) float64 time array [s]
    """
    N      = accel.shape[0]
    t_src  = np.arange(N) / src_rate
    t_up   = np.arange(0, t_src[-1], 1.0 / tgt_rate)

    accel_up = interp1d(t_src, accel, axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(accel[0], accel[-1]))(t_up)
    gyro_up  = interp1d(t_src, gyro,  axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(gyro[0], gyro[-1]))(t_up)
    return accel_up, gyro_up, t_up


def remove_gravity_body(accel_flu: np.ndarray, R_bn: np.ndarray) -> np.ndarray:
    """
    Remove gravity from body-frame accelerometer measurement.
    Tartan IMU paper Eq.1: a_input = accel_flu - R_bn @ [0, 0, -9.81]

    In FLU (Z-up), gravity in ENU is [0, 0, -9.81].
    R_bn maps navigation→body: g_body = R_bn @ [0, 0, -9.81].
    The accelerometer measures: accel_flu = a_kinematic + |g_body|
    Gravity-free: accel_flu - g_body = accel_flu - R_bn @ [0, 0, -9.81].

    Parameters
    ----------
    accel_flu : (N, 3) or (3,) — raw accelerometer in FLU
    R_bn      : (3, 3) — navigation-to-body rotation (R_bn = R_nb.T = Rbn.T)

    Returns
    -------
    a_input : same shape as accel_flu, gravity-free
    """
    g_enu   = np.array([0., 0., -9.81])
    g_body  = R_bn @ g_enu                  # gravity in body frame
    return accel_flu - g_body


def build_lstm_input(accel_up: np.ndarray, gyro_up: np.ndarray,
                     orient_src: np.ndarray, t_up: np.ndarray,
                     t_src: np.ndarray,
                     center_idx_up: int) -> np.ndarray:
    """
    Build the 10-step LSTM input tensor centred on a given timestep.

    The 10-step window covers [center_idx_up - 10*STEP_SAMPLES, center_idx_up].
    Each step is STEP_SAMPLES=200 samples (1 second at 200 Hz).

    Parameters
    ----------
    accel_up, gyro_up : (N_up, 3) — gravity-NOT-yet-removed (handled per-step below)
    orient_src        : (N_src, 3) — [roll, pitch, yaw] at src_rate for R_bn
    t_up, t_src       : time arrays for interpolation of orientation
    center_idx_up     : int — current index in 200-Hz arrays (exclusive end)

    Returns
    -------
    imu_lstm : (LSTM_STEPS, STEP_SAMPLES, 6) float32 or None if not enough data
    """
    start_idx = center_idx_up - LSTM_STEPS * STEP_SAMPLES
    if start_idx < 0:
        return None

    # Interpolate orientation from src_rate to tgt_rate for gravity removal
    roll_up  = interp1d(t_src, orient_src[:, 0], kind='linear',
                        bounds_error=False, fill_value='extrapolate')(t_up)
    pitch_up = interp1d(t_src, orient_src[:, 1], kind='linear',
                        bounds_error=False, fill_value='extrapolate')(t_up)
    yaw_up   = interp1d(t_src, orient_src[:, 2], kind='linear',
                        bounds_error=False, fill_value='extrapolate')(t_up)

    imu_lstm = np.zeros((LSTM_STEPS, STEP_SAMPLES, 6), dtype=np.float32)

    for step in range(LSTM_STEPS):
        s = start_idx + step * STEP_SAMPLES
        e = s + STEP_SAMPLES

        a_raw = accel_up[s:e]    # (200, 3)
        g_raw = gyro_up[s:e]     # (200, 3)

        # Use orientation at the middle of the step for gravity removal
        mid = (s + e) // 2
        roll_m, pitch_m, yaw_m = roll_up[mid], pitch_up[mid], yaw_up[mid]
        R_nb = _qto_Rbn(_qfrom_euler(roll_m, pitch_m, yaw_m))
        R_bn = R_nb.T

        a_gf = remove_gravity_body(a_raw, R_bn)   # (200, 3) gravity-free

        imu_lstm[step, :, 0:3] = a_gf.astype(np.float32)   # accel first
        imu_lstm[step, :, 3:6] = g_raw.astype(np.float32)  # gyro second

    return imu_lstm   # (10, 200, 6)


def gt_velocity_body(vel_enu: np.ndarray, orient: np.ndarray,
                     idx: int) -> np.ndarray:
    """
    Ground-truth velocity in body frame for training.
    v_body = R_bn @ v_enu  (R_bn = R_nb.T)

    Parameters
    ----------
    vel_enu : (N, 3)
    orient  : (N, 3) [roll, pitch, yaw]
    idx     : sample index

    Returns
    -------
    v_body : (3,) float64
    """
    R_nb  = _qto_Rbn(_qfrom_euler(orient[idx, 0], orient[idx, 1], orient[idx, 2]))
    R_bn  = R_nb.T
    return R_bn @ vel_enu[idx]


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _qnorm(q):
    n = np.linalg.norm(q); return q/n if n > 0. else np.array([1.,0.,0.,0.])

def _qfrom_euler(roll, pitch, yaw):
    cr,sr = np.cos(roll/2), np.sin(roll/2)
    cp,sp = np.cos(pitch/2), np.sin(pitch/2)
    cy,sy = np.cos(yaw/2), np.sin(yaw/2)
    return _qnorm(np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))

def _qto_Rbn(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
