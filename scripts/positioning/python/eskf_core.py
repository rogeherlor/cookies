import numpy as np
import pymap3d as pm
from math import sqrt

import ekf_core

# ---------------------------------------------------------------------------
# Quaternion utilities  (Hamilton convention, q = [w, x, y, z])
# q_NB  ≡  body-to-navigation rotation  (Solà notation: q^n_b)
# ---------------------------------------------------------------------------

def skew(v):
    """3×3 skew-symmetric matrix of vector v."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_mul(q1, q2):
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_from_axis_angle(dtheta):
    """Exact small-angle quaternion from a rotation vector dtheta."""
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return quat_normalize(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis = dtheta / angle
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])

def quat_to_rpy(q):
    """Convert q_NB to roll/pitch/yaw (ZYX Euler, ENU frame)."""
    w, x, y, z = q
    t0 = 2.0 * (w*x + y*z)
    t1 = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw])

def quat_from_euler(roll, pitch, yaw):
    """Build q_NB from ZYX Euler angles using Shepperd's method."""
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([qw, qx, qy, qz]))

def quat_to_Rbn(q):
    """Direction-cosine matrix R_NB (body to navigation)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)    ],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])

# ---------------------------------------------------------------------------
# Main EKF - Embedded Port Prototype (Position Only)
# ---------------------------------------------------------------------------

def run_ekf(nav_data, ekf_params=None, outage_config=None, use_3d_rotation=True):
    """
    Embedded-Optimized Error-State EKF.
    Strictly follows Solà's position-only updates with sparse matrix operations.
    """
    params = dict(ekf_core.DEFAULT_EKF_PARAMS)
    if ekf_params is not None:
        params.update(ekf_params)
    ekf_params = params

    accel_flu = nav_data.accel_flu
    gyro_flu  = nav_data.gyro_flu
    lla       = nav_data.lla
    orient    = nav_data.orient

    # We only grab vel_enu for initialization, it is ignored during the filter loop
    vel_enu   = nav_data.vel_enu 

    frecIMU = nav_data.sample_rate
    lla0    = nav_data.lla0

    g   = ekf_core.GRAVITY
    Ts  = 1.0 / frecIMU
    NN  = lla.shape[0]

    # GPS outage window
    if outage_config is None:
        A, B = 0, 0
    else:
        A = int(outage_config["start"] * frecIMU)
        B = int((outage_config["start"] + outage_config["duration"]) * frecIMU)

    # ------------------------------------------------------------------
    # HISTORICAL ARRAYS (For Python plotting ONLY. Do not use in C/C++)
    # ------------------------------------------------------------------
    p, v, r = np.zeros((NN, 3)), np.zeros((NN, 3)), np.zeros((NN, 3))
    bias_acc, bias_gyr = np.zeros((NN, 3)), np.zeros((NN, 3))
    std_pos, std_vel, std_orient = np.zeros((NN, 3)), np.zeros((NN, 3)), np.zeros((NN, 3))
    std_bias_acc, std_bias_gyr = np.zeros((NN, 3)), np.zeros((NN, 3))

    p[0, :] = pm.geodetic2enu(lla[0, 0], lla[0, 1], lla[0, 2], lla0[0], lla0[1], lla0[2])
    v[0, :] = vel_enu[0, :]
    r[0, :] = orient[0, :]

    # ------------------------------------------------------------------
    # CURRENT STATE VARIABLES (This is what lives in your embedded RAM)
    # ------------------------------------------------------------------
    pIMU = p[0, :].copy()
    vIMU = v[0, :].copy()
    q    = quat_from_euler(orient[0, 0], orient[0, 1], orient[0, 2])
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

    beta_acc = ekf_params["beta_acc"]
    beta_gyr = ekf_params["beta_gyr"]

    Q = np.zeros((15, 15))
    Q[3:6, 3:6]     = np.eye(3) * (ekf_params["Qvel"] * Ts**2)
    Q[6:9, 6:9]     = np.diag([ekf_params["QorientXY"], ekf_params["QorientXY"], ekf_params["QorientZ"]])
    Q[9:12, 9:12]   = np.eye(3) * (ekf_params["Qacc"] * Ts)
    Q[12:15, 12:15] = np.diag([ekf_params["QgyrXY"], ekf_params["QgyrXY"], ekf_params["QgyrZ"]]) * Ts

    P = np.diag([
        ekf_params["P_pos_std"], ekf_params["P_pos_std"], ekf_params["P_pos_std"],
        ekf_params["P_vel_std"], ekf_params["P_vel_std"], ekf_params["P_vel_std"],
        ekf_params["P_orient_std"], ekf_params["P_orient_std"], ekf_params["P_orient_std"] * 2,
        ekf_params["P_acc_std"], ekf_params["P_acc_std"], ekf_params["P_acc_std"],
        ekf_params["P_gyr_std"], ekf_params["P_gyr_std"], ekf_params["P_gyr_std"],
    ]) ** 2

    R_pos  = np.eye(3) * ekf_params["Rpos"]
    R_nhc  = np.eye(2) * ekf_params["Rnhc"]
    R_zupt = np.eye(3) * ekf_params["Rzupt"]

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    for i in range(0, NN - 1):

        # 1. Bias-corrected IMU measurements
        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # 2. Nominal-state propagation
        dtheta = omega_b * Ts
        q      = quat_normalize(quat_mul(q, quat_from_axis_angle(dtheta)))
        Rbn    = quat_to_Rbn(q)

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # 3. Error-state transition matrix F
        F = np.zeros((15, 15))
        F[0:3, 3:6]     = np.eye(3)
        F[3:6, 6:9]     = -Rbn @ skew(acc_b)
        F[3:6, 9:12]    = -Rbn                       
        F[6:9, 6:9]     = -skew(omega_b)              
        F[6:9, 12:15]   = -np.eye(3)                  
        F[9:12, 9:12]   = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + F * Ts
        Fd[6:9, 6:9] = quat_to_Rbn(quat_from_axis_angle(omega_b * Ts)).T

        # Prediction: Covariance Matrix (Only dense 15x15 operation)
        P  = Fd @ P @ Fd.T + Q
        # dx remains 0.0

        update_occurred = False

        # --------------------------------------------------------------
        # A. GPS Update (Position ONLY) - SPARSE
        # --------------------------------------------------------------
        gps_available = nav_data.gps_available[i]
        not_in_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_available and not_in_outage:
            # 3D Position Error 
            z_pos = np.array(pm.geodetic2enu(lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2])) - pIMU
            
            # Sparse Slicing eliminates H @ P @ H.T
            # S is 3x3, K is 15x3
            innov = z_pos - dx[0:3]
            S = P[0:3, 0:3] + R_pos
            K = P[:, 0:3] @ np.linalg.inv(S)

            dx = dx + K @ innov
            
            # Standard Covariance Update is much faster than Joseph form
            P = P - K @ S @ K.T
            P = 0.5 * (P + P.T)  # Force symmetry
            update_occurred = True

        # --------------------------------------------------------------
        # B. Non-Holonomic Constraint (NHC) - SPARSE
        # --------------------------------------------------------------
        if ekf_params["enable_nhc"]:
            Rnb    = Rbn.T
            v_body = Rnb @ vIMU

            z_nhc = -v_body[1:3] # Target lateral and vertical body speed to 0

            # H matrix blocks corresponding to velocity (dx[3:6]) and attitude (dx[6:9])
            H_v     = Rnb[1:3, :]
            H_theta = skew(v_body)[1:3, :]
            H_sparse = np.hstack((H_v, H_theta))  # 2x6 matrix

            # Innovation must account for any dx[3:9] accumulated from GPS update
            innov = z_nhc - (H_sparse @ dx[3:9])

            P_sparse = P[3:9, 3:9]
            S = H_sparse @ P_sparse @ H_sparse.T + R_nhc
            K = P[:, 3:9] @ H_sparse.T @ np.linalg.inv(S)

            dx = dx + K @ innov
            P  = P - K @ S @ K.T
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # --------------------------------------------------------------
        # C. Zero-Velocity Update (ZUPT) - SPARSE
        # --------------------------------------------------------------
        if ekf_params["enable_zupt"]:
            accel_dev = abs(np.linalg.norm(acc_b) - 9.81)
            gyro_mag  = np.linalg.norm(omega_b)
            speed     = np.linalg.norm(vIMU)

            if accel_dev < ekf_params["zupt_accel_threshold"] and gyro_mag < ekf_params["zupt_gyro_threshold"] and speed < 1.0:
                z_zupt = -vIMU
                innov  = z_zupt - dx[3:6] # Account for dx velocity

                S = P[3:6, 3:6] + R_zupt
                K = P[:, 3:6] @ np.linalg.inv(S)

                dx = dx + K @ innov
                P  = P - K @ S @ K.T
                P  = 0.5 * (P + P.T)
                update_occurred = True

        # --------------------------------------------------------------
        # D. Full Error Injection & Reset
        # --------------------------------------------------------------
        if update_occurred:
            pIMU += dx[0:3]
            vIMU += dx[3:6]
            b_a  += dx[9:12]
            b_g  += dx[12:15]

            delta_theta = dx[6:9]
            q = quat_normalize(quat_mul(q, quat_from_axis_angle(delta_theta)))

            # Covariance reset via G (Sola Eq 288)
            G           = np.eye(15)
            G[6:9, 6:9] = np.eye(3) - 0.5 * skew(delta_theta)
            P           = G @ P @ G.T

            # Completely zero the error state post-injection
            dx[:] = 0.0

        # --------------------------------------------------------------
        # Store outputs for Python Plotting 
        # --------------------------------------------------------------
        p[i + 1, :] = pIMU
        v[i + 1, :] = vIMU
        r[i + 1, :] = quat_to_rpy(q)
        bias_acc[i + 1, :] = b_a
        bias_gyr[i + 1, :] = b_g
        std_pos[i + 1, :]      = np.sqrt(np.diag(P[0:3, 0:3]))
        std_vel[i + 1, :]      = np.sqrt(np.diag(P[3:6, 3:6]))
        std_orient[i + 1, :]   = np.sqrt(np.diag(P[6:9, 6:9]))
        std_bias_acc[i + 1, :] = np.sqrt(np.diag(P[9:12, 9:12]))
        std_bias_gyr[i + 1, :] = np.sqrt(np.diag(P[12:15, 12:15]))

    return {
        "p": p, "v": v, "r": r,
        "bias_acc": bias_acc, "bias_gyr": bias_gyr,
        "std_pos": std_pos, "std_vel": std_vel, "std_orient": std_orient,
        "std_bias_acc": std_bias_acc, "std_bias_gyr": std_bias_gyr,
    }