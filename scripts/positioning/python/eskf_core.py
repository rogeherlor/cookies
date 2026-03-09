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
    """
    Exact small-angle quaternion from a rotation vector dtheta (Sola eq. 101).
    Returns q = [cos(|dtheta|/2),  sin(|dtheta|/2) * dtheta_hat].
    """
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
    """
    Build q_NB from ZYX Euler angles using Shepperd's method.
    """
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([qw, qx, qy, qz]))

def quat_to_Rbn(q):
    """Direction-cosine matrix R_NB (body to navigation) from q_NB. Sola eq. 22."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)    ],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])

# ---------------------------------------------------------------------------
# Main EKF
# ---------------------------------------------------------------------------

def run_ekf(nav_data, ekf_params=None, outage_config=None, use_3d_rotation=True):
    """
    Quaternion-based Error-State EKF for INS/GNSS integration.
    """
    params = dict(ekf_core.DEFAULT_EKF_PARAMS)
    if ekf_params is not None:
        params.update(ekf_params)
    ekf_params = params

    accel_flu = nav_data.accel_flu
    gyro_flu  = nav_data.gyro_flu
    vel_enu   = nav_data.vel_enu
    lla       = nav_data.lla
    orient    = nav_data.orient

    frecIMU = nav_data.sample_rate
    lla0    = nav_data.lla0

    g   = ekf_core.GRAVITY
    a   = ekf_core.WGS84_A
    e2  = ekf_core.WGS84_E2

    Ts = 1.0 / frecIMU

    # GPS outage window
    if outage_config is None:
        A, B = 0, 0
    else:
        t1 = outage_config["start"]
        d  = outage_config["duration"]
        A  = int(t1 * frecIMU)
        B  = int((t1 + d) * frecIMU)

    NN = lla.shape[0]

    p            = np.zeros((NN, 3))
    v            = np.zeros((NN, 3))
    r            = np.zeros((NN, 3))
    bias_acc     = np.zeros((NN, 3))
    bias_gyr     = np.zeros((NN, 3))
    std_pos      = np.zeros((NN, 3))
    std_vel      = np.zeros((NN, 3))
    std_orient   = np.zeros((NN, 3))
    std_bias_acc = np.zeros((NN, 3))
    std_bias_gyr = np.zeros((NN, 3))

    # Initialisation
    p[0, :] = pm.geodetic2enu(lla[0, 0], lla[0, 1], lla[0, 2], lla0[0], lla0[1], lla0[2])
    v[0, :] = vel_enu[0, :]
    r[0, :] = orient[0, :]

    pIMU = p[0, :].copy()
    vIMU = v[0, :].copy()

    roll0, pitch0, yaw0 = orient[0, :]
    q = quat_from_euler(roll0, pitch0, yaw0)

    # Add explicitly tracked nominal biases 
    b_a = np.zeros(3)
    b_g = np.zeros(3)

    beta_acc = ekf_params["beta_acc"]
    beta_gyr = ekf_params["beta_gyr"]

    # Error-state vector  dx = [dp, dv, dtheta, db_a, db_g]
    dx = np.zeros(15)

    # Process noise covariance Q (Assuming isotropic noise scaling)
    Q = np.zeros((15, 15))
    Q[3:6, 3:6]     = np.eye(3) * (ekf_params["Qvel"] * Ts**2)
    Q[6:9, 6:9]     = np.diag([ekf_params["QorientXY"], ekf_params["QorientXY"], ekf_params["QorientZ"]])
    Q[9:12, 9:12]   = np.eye(3) * (ekf_params["Qacc"] * Ts)
    Q[12:15, 12:15] = np.diag([ekf_params["QgyrXY"], ekf_params["QgyrXY"], ekf_params["QgyrZ"]]) * Ts

    R_pos = np.eye(3) * ekf_params["Rpos"]
    H_pos = np.zeros((3, 15))
    H_pos[0:3, 0:3] = np.eye(3)

    P = np.diag([
        ekf_params["P_pos_std"], ekf_params["P_pos_std"], ekf_params["P_pos_std"],
        ekf_params["P_vel_std"], ekf_params["P_vel_std"], ekf_params["P_vel_std"],
        ekf_params["P_orient_std"], ekf_params["P_orient_std"], ekf_params["P_orient_std"] * 2,
        ekf_params["P_acc_std"], ekf_params["P_acc_std"], ekf_params["P_acc_std"],
        ekf_params["P_gyr_std"], ekf_params["P_gyr_std"], ekf_params["P_gyr_std"],
    ]) ** 2

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for i in range(0, NN - 1):

        # Bias-corrected IMU measurements
        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # Nominal-state propagation
        dtheta = omega_b * Ts
        dq     = quat_from_axis_angle(dtheta)
        q      = quat_normalize(quat_mul(q, dq))
        Rbn    = quat_to_Rbn(q)

        if not use_3d_rotation:
            rpy = quat_to_rpy(q)
            q   = quat_from_euler(0.0, 0.0, rpy[2])
            Rbn = quat_to_Rbn(q)

        accENU = Rbn @ acc_b
        pIMU = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU = vIMU + Ts * (accENU + g)

        # Error-state transition matrix F
        F = np.zeros((15, 15))
        F[0:3, 3:6]   = np.eye(3)
        F[3:6, 6:9]   = -Rbn @ skew(acc_b)
        F[3:6, 9:12]  = -Rbn                       
        F[6:9, 6:9]   = -skew(omega_b)              
        F[6:9, 12:15] = -np.eye(3)                  
        F[9:12, 9:12]   = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + F * Ts
        
        # Exact closed-form formulation for attitude error transition matrix (Sola Eq 393)
        Fd[6:9, 6:9] = quat_to_Rbn(quat_from_axis_angle(omega_b * Ts)).T

        P  = Fd @ P @ Fd.T + Q
        dx = Fd @ dx # Remains identically 0 during blind prediction 

        update_occurred = False

        # --------------------------------------------------------------
        # GPS position update
        # --------------------------------------------------------------
        gps_available = nav_data.gps_available[i]
        not_in_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_available and not_in_outage:
            z_pos = np.array(pm.geodetic2enu(lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2])) - pIMU
            innov = z_pos - H_pos @ dx
            S     = H_pos @ P @ H_pos.T + R_pos
            K     = P @ H_pos.T @ np.linalg.inv(S)
            dx    = dx + K @ innov

            IKH = np.eye(15) - K @ H_pos
            P   = IKH @ P @ IKH.T + K @ R_pos @ K.T
            update_occurred = True

        # # --------------------------------------------------------------
        # # Non-holonomic constraint (NHC)
        # # --------------------------------------------------------------
        # if ekf_params["enable_nhc"]:
        #     Rnb    = Rbn.T
        #     v_body = Rnb @ vIMU

        #     z_nhc         = -v_body[1:3]          
        #     H_nhc         = np.zeros((2, 15))
        #     H_nhc[:, 3:6] = Rnb[1:3, :]
            
        #     # Correct Jacobian of velocity error to body orientation 
        #     H_nhc[:, 6:9] = skew(v_body)[1:3, :]

        #     innov   = z_nhc - H_nhc @ dx
        #     R_nhc   = np.eye(2) * ekf_params["Rnhc"]
        #     S_nhc   = H_nhc @ P @ H_nhc.T + R_nhc
        #     K_nhc   = P @ H_nhc.T @ np.linalg.inv(S_nhc)
        #     dx      = dx + K_nhc @ innov

        #     IKH_nhc = np.eye(15) - K_nhc @ H_nhc
        #     P       = IKH_nhc @ P @ IKH_nhc.T + K_nhc @ R_nhc @ K_nhc.T
        #     update_occurred = True

        # # --------------------------------------------------------------
        # # Zero-velocity update (ZUPT)
        # # --------------------------------------------------------------
        # if ekf_params["enable_zupt"]:
        #     accel_mag = np.linalg.norm(acc_b)
        #     accel_dev = abs(accel_mag - 9.81)
        #     gyro_mag  = np.linalg.norm(omega_b)
        #     speed     = np.linalg.norm(vIMU)

        #     is_stationary = (
        #         accel_dev < ekf_params["zupt_accel_threshold"]
        #         and gyro_mag < ekf_params["zupt_gyro_threshold"]
        #         and speed < 1.0
        #     )

        #     if is_stationary:
        #         z_zupt = -vIMU
        #         H_zupt = np.zeros((3, 15))
        #         H_zupt[0:3, 3:6] = np.eye(3)

        #         innov   = z_zupt - H_zupt @ dx
        #         R_zupt  = np.eye(3) * ekf_params["Rzupt"]
        #         S_zupt  = H_zupt @ P @ H_zupt.T + R_zupt
        #         K_zupt  = P @ H_zupt.T @ np.linalg.inv(S_zupt)
        #         dx      = dx + K_zupt @ innov

        #         IKH_zupt = np.eye(15) - K_zupt @ H_zupt
        #         P        = IKH_zupt @ P @ IKH_zupt.T + K_zupt @ R_zupt @ K_zupt.T
        #         update_occurred = True

        # # --------------------------------------------------------------
        # # Levelling constraint (roll, pitch -> 0)
        # # --------------------------------------------------------------
        # if ekf_params["enable_level"] and not use_3d_rotation:
        #     rpy     = quat_to_rpy(q)
        #     z_level = np.array([-rpy[0], -rpy[1]])

        #     H_level       = np.zeros((2, 15))
        #     H_level[0, 6] = 1.0
        #     H_level[1, 7] = 1.0

        #     innov    = z_level - H_level @ dx
        #     R_level  = np.eye(2) * ekf_params["Rlevel"]
        #     S_level  = H_level @ P @ H_level.T + R_level
        #     K_level  = P @ H_level.T @ np.linalg.inv(S_level)
        #     dx       = dx + K_level @ innov

        #     IKH_level = np.eye(15) - K_level @ H_level
        #     P         = IKH_level @ P @ IKH_level.T + K_level @ R_level @ K_level.T
        #     update_occurred = True

        # --------------------------------------------------------------
        # Full Error Injection & Reset (Solà Section 7.3)
        # --------------------------------------------------------------
        if update_occurred:
            pIMU += dx[0:3]
            vIMU += dx[3:6]

            delta_theta = dx[6:9]
            dq_corr     = quat_from_axis_angle(delta_theta)
            q           = quat_normalize(quat_mul(q, dq_corr))

            b_a  += dx[9:12]
            b_g  += dx[12:15]

            # Covariance reset via G (Sola Eq 288)
            G           = np.eye(15)
            G[6:9, 6:9] = np.eye(3) - 0.5 * skew(delta_theta)
            P           = G @ P @ G.T

            # Completely zero the error state post-injection
            dx[:] = 0.0

        # --------------------------------------------------------------
        # Store outputs
        # --------------------------------------------------------------
        p[i + 1, :] = pIMU
        v[i + 1, :] = vIMU
        r[i + 1, :] = quat_to_rpy(q)

        bias_acc[i + 1, :] = b_a
        bias_gyr[i + 1, :] = b_g

        std_pos[i + 1, :]      = np.sqrt(np.diag(P[0:3,   0:3  ]))
        std_vel[i + 1, :]      = np.sqrt(np.diag(P[3:6,   3:6  ]))
        std_orient[i + 1, :]   = np.sqrt(np.diag(P[6:9,   6:9  ]))
        std_bias_acc[i + 1, :] = np.sqrt(np.diag(P[9:12,  9:12 ]))
        std_bias_gyr[i + 1, :] = np.sqrt(np.diag(P[12:15, 12:15]))

    return {
        "p":            p,
        "v":            v,
        "r":            r,
        "bias_acc":     bias_acc,
        "bias_gyr":     bias_gyr,
        "std_pos":      std_pos,
        "std_vel":      std_vel,
        "std_orient":   std_orient,
        "std_bias_acc": std_bias_acc,
        "std_bias_gyr": std_bias_gyr,
    }