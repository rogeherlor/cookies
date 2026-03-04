# -*- coding: utf-8 -*-
"""
Core EKF implementation - shared between ekf.py and ekf_genetic.py
"""
import numpy as np
from math import sin, cos, tan, radians, sqrt
import pymap3d as pm


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
        if use_3d_rotation:
            # Full 3D rotation for FLU (Forward-Left-Up) body frame, ZYX Euler sequence.
            # Ry uses -sin in [0,2] / +sin in [2,0] because the FLU pitch axis (y) points
            # left, opposite to the FRD aerospace convention where Ry has +sin in [0,2].
            Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
            Ry = np.array([[cos(pitch), 0, -sin(pitch)], [0, 1, 0], [sin(pitch), 0, cos(pitch)]])
            Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
            Rbn = Rz @ Ry @ Rx  # ZYX Euler angle sequence
        else:
            # 2D rotation (yaw only - simpler for planar motion)
            Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
            Rbn = Rz
        
        accENU = Rbn @ acc
        
        pIMU = pIMU + Ts * vIMU + Ts**2 / 2 * (accENU + g)
        vIMU = vIMU + Ts * (accENU + g)
        # Euler Rate Transformation Matrix (W) to relate Angular Velocity to Euler Angle Rates
        W = np.array([
            [1, sin(roll) * tan(pitch), cos(roll) * tan(pitch)],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll) / cos(pitch), cos(roll) / cos(pitch)]
        ])
        rpy = rpy + Ts * W @ gyr
        
        # Wrap yaw to [-π, π]
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
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3, 7] = fU
        F[3, 8] = -fN
        F[4, 6] = -fU
        F[4, 8] = fE
        F[5, 6] = fN
        F[5, 7] = -fE
        
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
            rpy[0] += x[6]
            rpy[1] += x[7]
            rpy[2] += x[8]
        
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
        
        # Error state reset after update (Error-State EKF)
        if gps_is_available and not_in_outage:
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
