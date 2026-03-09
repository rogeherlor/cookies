# -*- coding: utf-8 -*-
"""
IMU-Only — Pure inertial dead reckoning without any GNSS aiding.

This baseline uses the same quaternion mechanization as the ESKF but
applies no Kalman updates at all.  Errors accumulate unbounded from
sensor noise and bias drift.  Its purpose is to show how much the
Kalman filters reduce drift, so it should NOT be used to draw performance
conclusions — it simply establishes the "no-filter" lower bound.

Bias is initialised to zero and never corrected.

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Quaternion: Hamilton convention  q = [w, x, y, z]  (q_NB = body → nav)
"""
import numpy as np
import pymap3d as pm

GRAVITY = np.array([0.0, 0.0, -9.81])   # ENU [m/s²]

DEFAULT_PARAMS = {}   # no tunable parameters for pure dead reckoning


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qfrom_axis_angle(dtheta):
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return _qnorm(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis = dtheta / angle;  s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0]*s, axis[1]*s, axis[2]*s])


def _qfrom_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    return _qnorm(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]))


def _qto_rpy(q):
    w, x, y, z = q
    roll  = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def _qto_Rbn(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)    ],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)    ],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


# ── Main function ──────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Pure IMU dead reckoning (no GNSS, no filter, no bias estimation).

    outage_config is accepted for interface compatibility but ignored —
    this function never uses GNSS regardless.

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
        All std arrays are zeros (no covariance tracked).
    """
    accel_flu = nav_data.accel_flu
    gyro_flu  = nav_data.gyro_flu
    lla       = nav_data.lla
    orient    = nav_data.orient
    vel_enu   = nav_data.vel_enu
    frecIMU   = nav_data.sample_rate
    lla0      = nav_data.lla0

    g  = GRAVITY
    Ts = 1.0 / frecIMU
    NN = lla.shape[0]

    pos     = np.zeros((NN, 3))
    vel     = np.zeros((NN, 3))
    rpy_out = np.zeros((NN, 3))

    pos[0, :]     = pm.geodetic2enu(lla[0,0], lla[0,1], lla[0,2], lla0[0], lla0[1], lla0[2])
    vel[0, :]     = vel_enu[0, :]
    rpy_out[0, :] = orient[0, :]

    pIMU = pos[0, :].copy()
    vIMU = vel[0, :].copy()
    q    = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])

    for i in range(NN - 1):
        # Raw IMU (no bias correction — bias stays zero throughout)
        acc_b   = accel_flu[i, :]
        omega_b = gyro_flu[i, :]

        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0.0, 0.0, omega_b[2] * Ts])

        q   = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn = _qto_Rbn(q)

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        pos[i+1, :]     = pIMU
        vel[i+1, :]     = vIMU
        rpy_out[i+1, :] = _qto_rpy(q)

    # Zero arrays for all filter-specific outputs (no covariance, no bias estimation)
    zeros = np.zeros((NN, 3))
    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': zeros.copy(), 'bias_gyr': zeros.copy(),
        'std_pos': zeros.copy(), 'std_vel': zeros.copy(), 'std_orient': zeros.copy(),
        'std_bias_acc': zeros.copy(), 'std_bias_gyr': zeros.copy(),
    }
