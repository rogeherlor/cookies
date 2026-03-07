# -*- coding: utf-8 -*-
"""
Core EKF implementation - shared between ekf.py and ekf_genetic.py
"""
import numpy as np
from math import sin, cos, tan, radians, sqrt
import pymap3d as pm

def skew(v):
    """Return the skew-symmetric matrix of vector v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# ─── Physical constants (WGS-84 ellipsoid) ────────────────────────────────────
GRAVITY = np.array([0, 0, -9.81])        # gravity vector in ENU frame [m/s²]
WGS84_A = 6378137.0                       # semi-major axis (equatorial radius) [m]
WGS84_B = 6356752.3142                    # semi-minor axis (polar radius) [m]
WGS84_E2 = 0.00669437999                  # first eccentricity squared  (1 - b²/a²)
WGS84_EP2 = 0.00673949674                 # second eccentricity squared (a²/b² - 1)


# ─── Default tunable EKF parameters ──────────────────────────────────────────
DEFAULT_EKF_PARAMS = {
    # Process noise Q
    'Qpos': 5.312e-06,
    'Qvel': 4.702e-06,
    'QorientXY': 0.0002,
    'QorientZ': 0.2,
    'Qacc': 0.1,
    'QgyrXY': 0.0001,
    'QgyrZ': 0.1,

    # Measurement noise R
    'Rpos': 67.79,

    # Gauss-Markov time constants (negative for stable 1st-order process)
    'beta_acc': -1.910e-06,
    'beta_gyr': -7.077e-02,

    # Initial covariance P (standard deviations)
    'P_pos_std': 0.23,
    'P_vel_std': 0.17,
    'P_orient_std': 0.239,
    'P_acc_std': 0.01,
    'P_gyr_std': 0.001,

    # Non-Holonomic Constraints (NHC)
    # Cars can't slide sideways or fly: v_lateral ≈ 0, v_vertical ≈ 0 in body frame
    'enable_nhc': True,
    'Rnhc': 0.1,              # NHC measurement noise (m/s)² — how much lateral/vertical slip to tolerate

    # Zero Velocity Updates (ZUPT)
    # When vehicle is stationary: v ~ 0 in navigation frame
    'enable_zupt': True,
    'Rzupt': 0.01,            # ZUPT measurement noise (m/s)^2 -- very tight since truly stopped
    'zupt_accel_threshold': 0.3,  # Specific-force deviation from gravity (m/s^2) to detect standstill
    'zupt_gyro_threshold': 0.05,  # Gyro magnitude threshold (rad/s) to detect standstill

    # Leveling constraint (2D mode only)
    # On flat ground roll ~ 0 and pitch ~ 0; inject as pseudo-measurement every epoch.
    # Only active when use_3d_rotation=False (2D mode), since in 3D the vehicle
    # may actually have non-zero roll/pitch.
    'enable_level': True,
    'Rlevel': 0.001,          # Leveling measurement noise (rad)^2 -- tight for flat roads
}


def run_ekf(nav_data, ekf_params=None, outage_config=None, use_3d_rotation=True):
    """
    Run Error-State Extended Kalman Filter for INS/GNSS integration.
    
    Args:
        nav_data: Navigation dataset with accel_flu, gyro_flu, vel_enu, lla, orient, etc.
        ekf_params: Dictionary with EKF parameters:
            - Qpos, Qvel, QorientXY, QorientZ, Qacc, QgyrXY, QgyrZ (process noise)
            - Rpos (measurement noise)
            - beta_acc, beta_gyr (Gauss-Markov coefficients)
            - P_pos_std, P_vel_std, P_orient_std, P_acc_std, P_gyr_std (initial covariance)
            - enable_nhc, Rnhc (Non-Holonomic Constraints)
            - enable_zupt, Rzupt, zupt_accel_threshold, zupt_gyro_threshold (Zero Velocity Updates)
            - enable_level, Rlevel (Leveling constraint: roll/pitch ~ 0 in 2D mode)
        outage_config: Optional dict with 'start' (seconds) and 'duration' (seconds) for GPS outage
        use_3d_rotation: If True, use full 3D rotation (roll/pitch/yaw). If False, use 2D rotation (yaw only).
    
    Returns:
        Dictionary with:
            - p: Estimated position trajectory [N×3] ENU
            - v: Estimated velocity trajectory [N×3] ENU
            - r: Estimated orientation trajectory [N×3] roll/pitch/yaw
            - bias_acc: Accelerometer bias estimates [N×3]
            - bias_gyr: Gyroscope bias estimates [N×3]
            - std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr: Uncertainties
    """
    # Merge caller overrides on top of defaults
    params = dict(DEFAULT_EKF_PARAMS)
    if ekf_params is not None:
        params.update(ekf_params)
    ekf_params = params

    # Extract data arrays
    accel_flu = nav_data.accel_flu
    gyro_flu = nav_data.gyro_flu
    vel_enu = nav_data.vel_enu
    lla = nav_data.lla
    orient = nav_data.orient
    
    frecIMU = nav_data.sample_rate
    lla0 = nav_data.lla0
    
    # Alias module-level constants for readability
    g  = GRAVITY
    a  = WGS84_A
    e2 = WGS84_E2
    
    Ts = 1 / frecIMU
    
    # GPS outage configuration
    if outage_config is None:
        A, B = 0, 0
    else:
        t1 = outage_config['start']
        d = outage_config['duration']
        A = int(t1 * frecIMU)
        B = int((t1 + d) * frecIMU)
    
    # Output storage
    NN = lla.shape[0]
    p = np.zeros((NN, 3))
    v = np.zeros((NN, 3))
    r = np.zeros((NN, 3))
    bias_acc = np.zeros((NN, 3))
    bias_gyr = np.zeros((NN, 3))
    std_pos = np.zeros((NN, 3))
    std_vel = np.zeros((NN, 3))
    std_orient = np.zeros((NN, 3))
    std_bias_acc = np.zeros((NN, 3))
    std_bias_gyr = np.zeros((NN, 3))
    
    # First state
    p[0, :] = pm.geodetic2enu(lla[0, 0], lla[0, 1], lla[0, 2], lla0[0], lla0[1], lla0[2])
    v[0, :] = vel_enu[0, :]
    r[0, :] = orient[0, :]
    
    # Navigation State Initialisation
    pIMU = p[0, :].T
    vIMU = v[0, :].T
    rpy = np.zeros(3)  # Unknown attitude [roll, pitch, yaw]
    
    # Filter Initialisation
    beta_acc = ekf_params['beta_acc']
    beta_gyr = ekf_params['beta_gyr']
    
    # State initialisation x = (δp[3], δv[3], δϵ[3], b_acc[3], b_gyr[3])
    x = np.zeros(15)
    x[9:12] = 1e-6  # Small initial bias
    x[12:15] = 1e-7
    
    # Process noise Q
    Q = np.diag([ekf_params['Qpos']] * 3 + 
                [ekf_params['Qvel']] * 3 + 
                [ekf_params['QorientXY'], ekf_params['QorientXY'], ekf_params['QorientZ']] + 
                [ekf_params['Qacc']] * 3 + 
                [ekf_params['QgyrXY'], ekf_params['QgyrXY'], ekf_params['QgyrZ']])
    
    # Measurement noise R
    R = np.eye(3) * ekf_params['Rpos']
    
    # Observation matrix H
    H = np.zeros((3, 15))
    H[0:3, 0:3] = np.eye(3)
    
    # Initial covariance P
    P = np.diag([
        ekf_params['P_pos_std'], ekf_params['P_pos_std'], ekf_params['P_pos_std'],
        ekf_params['P_vel_std'], ekf_params['P_vel_std'], ekf_params['P_vel_std'],
        ekf_params['P_orient_std'], ekf_params['P_orient_std'], ekf_params['P_orient_std'] * 2,
        ekf_params['P_acc_std'], ekf_params['P_acc_std'], ekf_params['P_acc_std'],
        ekf_params['P_gyr_std'], ekf_params['P_gyr_std'], ekf_params['P_gyr_std']
    ]) ** 2
    
    K = np.zeros((15, 3))
    
    # Main EKF loop
    for i in range(0, NN - 1):
        # IMU Correction - subtract estimated biases
        acc = accel_flu[i, :] - x[9:12]
        gyr = gyro_flu[i, :] - x[12:15]
        
        # Navigation equations - IMU State Estimation
        roll, pitch, yaw = rpy
        
        # Rotation matrix (Body to Navigation frame)
        #   Rbn = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (ZYX Euler sequence, FLU body)
        cr, sr_ = cos(roll), sin(roll)
        cp, sp  = cos(pitch), sin(pitch)
        cy, sy  = cos(yaw), sin(yaw)

        Rz_mat = np.array([[ cy, -sy, 0], [ sy, cy, 0], [0, 0, 1]])
        Ry_mat = np.array([[ cp, 0, -sp], [  0, 1,  0], [sp, 0, cp]])
        Rx_mat = np.array([[  1, 0,   0], [  0, cr, -sr_], [0, sr_, cr]])

        if use_3d_rotation:
            Rbn = Rz_mat @ Ry_mat @ Rx_mat
        else:
            # 2D rotation (yaw only - simpler for planar motion)
            Rbn = Rz_mat
        
        accENU = Rbn @ acc
        
        pIMU = pIMU + Ts * vIMU + Ts**2 / 2 * (accENU + g)
        vIMU = vIMU + Ts * (accENU + g)
        # Euler Rate Transformation Matrix (W) to relate Angular Velocity to Euler Angle Rates
        tp = tan(pitch)
        W = np.array([
            [1, sr_ * tp, cr * tp],
            [0, cr,       -sr_],
            [0, sr_ / cp,  cr / cp]
        ])
        rpy = rpy + Ts * W @ gyr
        
        # Wrap yaw to [-pi, pi]
        rpy[2] = (rpy[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Prediction
        fE, fN, fU = accENU
        
        # Compute radii of curvature for orientation dynamics
        llaIMU = pm.enu2geodetic(pIMU[0], pIMU[1], pIMU[2], lla0[0], lla0[1], lla0[2])
        lat_rad = radians(llaIMU[0])
        alt_imu = llaIMU[2]
        sin_lat = sin(lat_rad)
        M = a * (1 - e2) / (1 - e2 * sin_lat**2)**1.5  # Meridional radius
        N_radius = a / sqrt(1 - e2 * sin_lat**2)        # Prime vertical radius
        
        # Dynamic matrix F
        #
        # Velocity–orientation coupling  F[3:6, 6:9]
        # ---------------------------------------------------------
        # The error-state uses Euler-angle perturbations (δroll, δpitch, δyaw),
        # so the correct Jacobian is  d(Rbn·a_body)/d(roll, pitch, yaw):
        #
        #   col 0 (∂/∂roll)  = Rz · Ry · dRx/droll  · a_body
        #   col 1 (∂/∂pitch) = Rz · dRy/dpitch · Rx · a_body
        #   col 2 (∂/∂yaw)   = dRz/dyaw · Ry · Rx   · a_body
        #
        # Note: the ±skew(f) formula from Sola/Groves is for rotation-vector
        # error states, NOT Euler-angle error states.  Using it here causes a
        # sign error that creates positive feedback ⇒ pitch divergence.
        #
        # IMPORTANT: In 2D mode the mechanization uses Rbn = Rz (yaw only),
        # so accENU = Rz @ acc.  The Jacobian MUST be consistent:
        #   d(Rz·acc)/d(roll)  = 0      (Rz does not depend on roll)
        #   d(Rz·acc)/d(pitch) = 0      (Rz does not depend on pitch)
        #   d(Rz·acc)/d(yaw)   = dRz · acc
        # Using the full 3D derivatives in 2D mode creates a mismatch:
        # the filter thinks roll/pitch affect position, but they don't
        # in the actual dynamics => GPS corrections inject noise into
        # roll/pitch to explain position errors unrelated to them.
        # ---------------------------------------------------------
        dRx = np.array([[ 0,   0,    0],
                        [ 0, -sr_, -cr],
                        [ 0,  cr,  -sr_]])
        dRy = np.array([[-sp, 0, -cp],
                        [  0, 0,   0],
                        [ cp, 0, -sp]])
        dRz = np.array([[-sy, -cy, 0],
                        [ cy, -sy, 0],
                        [  0,   0, 0]])

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        # Each column is the derivative of accENU w.r.t. one Euler angle
        if use_3d_rotation:
            F[3:6, 6] = Rz_mat @ Ry_mat @ dRx @ acc   # d(accENU)/d(roll)
            F[3:6, 7] = Rz_mat @ dRy @ Rx_mat @ acc   # d(accENU)/d(pitch)
            F[3:6, 8] = dRz @ Ry_mat @ Rx_mat @ acc   # d(accENU)/d(yaw)
        else:
            # 2D mode: Rbn = Rz, so only yaw column is nonzero
            # F[3:6, 6] = 0  (roll doesn't affect Rz·acc)
            # F[3:6, 7] = 0  (pitch doesn't affect Rz·acc)
            F[3:6, 8] = dRz @ acc                      # d(Rz·acc)/d(yaw)
        
        # Orientation error dynamics (Earth curvature / transport rate coupling)
        # Transport rate: ω_en = [vN/(M+h), -vE/(N+h), -vE·tan(lat)/(N+h)]
        F[6, 4] =  1.0 / (M + alt_imu)                   # ∂ω_E/∂vN  → roll rate error
        F[7, 3] = -1.0 / (N_radius + alt_imu)            # ∂ω_N/∂vE  → pitch rate error
        F[8, 3] = -tan(lat_rad) / (N_radius + alt_imu)   # ∂ω_U/∂vE  → yaw rate error
        
        F[3:6, 9:12] = -Rbn
        F[6:9, 12:15] = -W    # Euler-angle error state: gyro bias couples via -W, not -Rbn
        F[9:12, 9:12] = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)
        
        F = np.eye(15) + F * Ts
        P = F @ P @ F.T + Q
        x = F @ x
        
        # Update - check if GPS is available and not in outage window
        gps_is_available = nav_data.gps_available[i]
        not_in_outage = ((i + 1) < A or (i + 1) > B)
        
        if gps_is_available and not_in_outage:
            # GPS measurement is available - perform Kalman update
            z = pm.geodetic2enu(lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]) - pIMU
            innovation = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ innovation
            
            # Joseph-form covariance update (numerically stable, preserves symmetry)
            IKH = np.eye(15) - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T
            
            # Inject error state into nominal state
            pIMU += x[0:3]
            vIMU += x[3:6]
            
            delta_theta = x[6:9]
            if np.linalg.norm(delta_theta) > 1e-10:
                Rbn = (np.eye(3) - skew(delta_theta)) @ Rbn
                pitch = np.arcsin(np.clip(-Rbn[2, 0], -1.0, 1.0))
                roll = np.arctan2(Rbn[2, 1], Rbn[2, 2])
                yaw = np.arctan2(Rbn[1, 0], Rbn[0, 0])
                rpy = np.array([roll, pitch, yaw])
                
            # Reset injected states so downstream updates (NHC/ZUPT/Level)
            # start from a clean error state and don't double-count
            x[0:9] = 0
        
        # ── Non-Holonomic Constraint (NHC) update ────────────────────────
        # A car cannot slide sideways or fly: body-frame lateral and vertical
        # velocity should be ~0.  We express this as a pseudo-measurement:
        #   h(x) = Rbn^T * v_ENU  -> [v_lateral, v_vertical] = [0, 0]
        #
        # Linearisation for the error-state EKF:
        #   delta_v_body = Rnb * delta_v  -  Rnb * [v_ENU]x * delta_eps
        # The negative sign arises because the true rotation is
        #   R_true = (I + [delta_eps]x) * R_nominal
        # so  R_true^T = R_nominal^T * (I - [delta_eps]x)
        # =>  v_body_true = Rnb*v - Rnb*[v]x*delta_eps + Rnb*delta_v
        if ekf_params['enable_nhc']:
            # Recompute Rbn from current (possibly GPS-corrected) rpy
            roll_c, pitch_c, yaw_c = rpy
            if use_3d_rotation:
                Rz_c = np.array([[cos(yaw_c), -sin(yaw_c), 0], [sin(yaw_c), cos(yaw_c), 0], [0, 0, 1]])
                Ry_c = np.array([[cos(pitch_c), 0, -sin(pitch_c)], [0, 1, 0], [sin(pitch_c), 0, cos(pitch_c)]])
                Rx_c = np.array([[1, 0, 0], [0, cos(roll_c), -sin(roll_c)], [0, sin(roll_c), cos(roll_c)]])
                Rbn_c = Rz_c @ Ry_c @ Rx_c
            else:
                Rz_c = np.array([[cos(yaw_c), -sin(yaw_c), 0], [sin(yaw_c), cos(yaw_c), 0], [0, 0, 1]])
                Rbn_c = Rz_c

            Rnb_c = Rbn_c.T  # Navigation-to-body rotation
            v_body = Rnb_c @ vIMU  # Velocity in body frame [forward, lateral, vertical]

            # Measurement: lateral and vertical body velocity should be zero
            z_nhc = -v_body[1:3]  # innovation = 0 - v_body(lateral, vertical)

            # Observation matrix: H_nhc maps error state -> body-frame velocity error
            H_nhc = np.zeros((2, 15))
            H_nhc[:, 3:6] = Rnb_c[1:3, :]  # dv_body / d(delta_v)

            # Skew-symmetric matrix of vIMU for cross-product [vIMU]x
            vE, vN, vU = vIMU
            skew_v = np.array([[0, -vU, vN],
                               [vU, 0, -vE],
                               [-vN, vE, 0]])
            # Negative sign: see derivation above
            H_nhc[:, 6:9] = -(Rnb_c @ skew_v)[1:3, :]  # dv_body / d(delta_eps)

            R_nhc = np.eye(2) * ekf_params['Rnhc']
            S_nhc = H_nhc @ P @ H_nhc.T + R_nhc
            K_nhc = P @ H_nhc.T @ np.linalg.inv(S_nhc)
            x = x + K_nhc @ (z_nhc - H_nhc @ x)

            IKH_nhc = np.eye(15) - K_nhc @ H_nhc
            P = IKH_nhc @ P @ IKH_nhc.T + K_nhc @ R_nhc @ K_nhc.T

            # Inject NHC corrections into nominal state
            vIMU += x[3:6]
            
            delta_theta = x[6:9]
            if np.linalg.norm(delta_theta) > 1e-10:
                Rbn = (np.eye(3) - skew(delta_theta)) @ Rbn
                pitch = np.arcsin(np.clip(-Rbn[2, 0], -1.0, 1.0))
                roll = np.arctan2(Rbn[2, 1], Rbn[2, 2])
                yaw = np.arctan2(Rbn[1, 0], Rbn[0, 0])
                rpy = np.array([roll, pitch, yaw])
                
            x[3:9] = 0  # Reset corrected error states

        # ── Zero Velocity Update (ZUPT) ──────────────────────────────────
        # Detect standstill from IMU: if specific force ~ gravity and gyro ~ 0,
        # the vehicle is stationary -> inject v = [0,0,0] as measurement.
        #
        # Detection uses THREE conditions to avoid false triggers during
        # constant-velocity cruising (where accel ~ gravity and gyro ~ 0 too):
        #   1. Specific-force magnitude ~ 9.81  (no dynamic acceleration)
        #   2. Gyro magnitude ~ 0               (no rotation)
        #   3. Current estimated speed ~ 0      (not cruising)
        if ekf_params['enable_zupt']:
            accel_mag = np.linalg.norm(acc)
            accel_dev = abs(accel_mag - 9.81)
            gyro_mag = np.linalg.norm(gyr)
            speed = np.linalg.norm(vIMU)

            is_stationary = (accel_dev < ekf_params['zupt_accel_threshold'] and
                             gyro_mag < ekf_params['zupt_gyro_threshold'] and
                             speed < 1.0)  # m/s - must already be near-stopped

            if is_stationary:
                # Measurement: velocity in navigation frame = 0
                z_zupt = -vIMU  # innovation = 0 - vIMU

                H_zupt = np.zeros((3, 15))
                H_zupt[0:3, 3:6] = np.eye(3)  # Observes δv directly

                R_zupt = np.eye(3) * ekf_params['Rzupt']
                S_zupt = H_zupt @ P @ H_zupt.T + R_zupt
                K_zupt = P @ H_zupt.T @ np.linalg.inv(S_zupt)
                x = x + K_zupt @ (z_zupt - H_zupt @ x)

                IKH_zupt = np.eye(15) - K_zupt @ H_zupt
                P = IKH_zupt @ P @ IKH_zupt.T + K_zupt @ R_zupt @ K_zupt.T

                # Inject ZUPT corrections
                vIMU += x[3:6]
                x[3:6] = 0  # Reset velocity error state

        # -- Leveling constraint (roll ~ 0, pitch ~ 0) -----------------------
        # In 2D mode the vehicle is assumed to stay on flat ground, so
        # roll and pitch are observable as ~0.  This directly constrains
        # the orientation error states delta_eps_roll and delta_eps_pitch,
        # preventing gravity-projection drift that dominates position error.
        if ekf_params['enable_level'] and not use_3d_rotation:
            # Measurement: roll and pitch should be zero
            z_level = np.array([-rpy[0], -rpy[1]])  # innovation = 0 - [roll, pitch]

            H_level = np.zeros((2, 15))
            H_level[0, 6] = 1.0  # observes delta_eps_roll
            H_level[1, 7] = 1.0  # observes delta_eps_pitch

            R_level = np.eye(2) * ekf_params['Rlevel']
            S_level = H_level @ P @ H_level.T + R_level
            K_level = P @ H_level.T @ np.linalg.inv(S_level)
            x = x + K_level @ (z_level - H_level @ x)

            IKH_level = np.eye(15) - K_level @ H_level
            P = IKH_level @ P @ IKH_level.T + K_level @ R_level @ K_level.T

            # Inject leveling corrections
            delta_theta_level = x[6:9]
            if np.linalg.norm(delta_theta_level) > 1e-10:
                Rbn = (np.eye(3) - skew(delta_theta_level)) @ Rbn
                pitch = np.arcsin(np.clip(-Rbn[2, 0], -1.0, 1.0))
                roll = np.arctan2(Rbn[2, 1], Rbn[2, 2])
                yaw = np.arctan2(Rbn[1, 0], Rbn[0, 0])
                rpy = np.array([roll, pitch, yaw])
                
            x[6:9] = 0  # Reset orientation error states

        # Store current iteration results
        p[i + 1, :] = pIMU.T
        v[i + 1, :] = vIMU.T
        r[i + 1, 0] = rpy[0]
        r[i + 1, 1] = rpy[1]
        r[i + 1, 2] = rpy[2]
        
        # Store bias estimates
        bias_acc[i + 1, :] = x[9:12]
        bias_gyr[i + 1, :] = x[12:15]
        
        # Store uncertainty
        std_pos[i + 1, :] = np.sqrt(np.diag(P[0:3, 0:3]))
        std_vel[i + 1, :] = np.sqrt(np.diag(P[3:6, 3:6]))
        std_orient[i + 1, :] = np.sqrt(np.diag(P[6:9, 6:9]))
        std_bias_acc[i + 1, :] = np.sqrt(np.diag(P[9:12, 9:12]))
        std_bias_gyr[i + 1, :] = np.sqrt(np.diag(P[12:15, 12:15]))
        
        # Error state reset after storing (Error-State EKF)
        # Position/velocity/orientation error states are reset every step
        # because the nominal state always absorbs them. Bias states
        # (x[9:14]) persist and are estimated cumulatively.
        x[0:9] = 0
    
    return {
        'p': p,
        'v': v,
        'r': r,
        'bias_acc': bias_acc,
        'bias_gyr': bias_gyr,
        'std_pos': std_pos,
        'std_vel': std_vel,
        'std_orient': std_orient,
        'std_bias_acc': std_bias_acc,
        'std_bias_gyr': std_bias_gyr
    }
