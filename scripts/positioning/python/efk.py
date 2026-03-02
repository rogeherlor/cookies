import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, tan, radians, sqrt
import pymap3d as pm
from pathlib import Path
from datetime import datetime
import metrics
import data_loader

################### LOAD DATA ###########################
# Load navigation data from KITTI dataset
nav_data = data_loader.get_kitti_dataset('10_03_0034')

# Extract data arrays
accel_flu = nav_data.accel_flu
gyro_flu = nav_data.gyro_flu
vel_enu = nav_data.vel_enu
lla = nav_data.lla
orient = nav_data.orient

########################## VAR INIT ###############
# No GNSS loss: t1=0, d=0
t1 = 20  # time init loss in seconds
d = 10   # duration of GNSS loss in seconds
frecIMU = nav_data.sample_rate  # Use sample rate from data
frecGPS = 1                      # 1 Hz

lla0 = np.array([49.01,8.43,116.43])    # origin of ENU coordinate system
g = np.array([0,0,-9.81])               # gravity acceleration
a = 6378137.0                           # equatorial radius [m]
b = 6356752.3142                        # polar radius [m]
e2 = 0.00669437999                      # eccentricity^2 (1-b^2/a^2)
ep2 = 0.00673949674                     # second eccentricity^2 (a^2/b^2 -1)

t2 = t1+d                               # time end loss in seconds
A = int(t1*frecIMU)                     # data index start loss
B = int(t2*frecIMU)                     # data index end loss
Ts= 1/frecIMU                           # sampling time

# Output Storage Navigation State Estimates
NN = lla.shape[0]   # number of data points in the trajectory
p = np.zeros((NN,3))
v = np.zeros((NN,3))
r = np.zeros((NN,3))
# First state
p[0,:] = pm.geodetic2enu(lla[0,0],lla[0,1],lla[0,2],lla0[0],lla0[1],lla0[2])
v[0,:] = vel_enu[0,:]
r[0,:] = orient[0,:]

# Navigation State Initialisation
pIMU = p[0,:].T     # known position
vIMU = v[0,:].T     # known velocity
rpy = np.zeros(3)   # unknown attitude [roll, pitch, yaw]

## Filter Initialisation
beta_acc = -3.7e-7 # Gauss-Markov coef # TODO: get from modelling
beta_gyr = -2.9e-1 # Gauss-Markov coef # TODO: get from modelling

# State initialisation x
accBias = 1e-6      # bias inicial
gyrBias = 1e-7      # bias inicial
x = np.zeros(15)    # state x = (δp[3], δv[3], δϵ[3], b_acc[3], b_gyr[3])
                    # δϵ = [δφ, δθ, δψ] = [δroll, δpitch, δyaw] (indices 6,7,8)
x[9:12] = accBias
x[12:15] = gyrBias

# Process noise Q # TODO: get from genetic
Qpos, Qvel = 2, 2
QorientX, QorientY, QorientZ = 0.0002, 0.0002, 0.2
Qacc = 0.1
QgyrX, QgyrY, QgyrZ = 0.0001, 0.0001, 0.1
Q = np.diag([Qpos]*3 + [Qvel]*3 + [QorientX, QorientY, QorientZ] + [Qacc]*3 + [QgyrX, QgyrY, QgyrZ])

# Measurement noise R # TODO: get from genetic
Rpos = 2
R = np.eye(3)*Rpos

# Observation matrix H
H = np.zeros((3,15))
H[0:3,0:3]=np.eye(3)

# Initial state covariance matrix (P) and Kalman gain (K)
P = np.eye(15)*0.0 # TODO: get from modelling
K = np.zeros((15,3))

################## RUN #########################
for i in range(0,NN-1):
    
    # IMU Correction - subtract estimated biases from measurements
    acc = accel_flu[i,:] - x[9:12]
    gyr = gyro_flu[i,:] - x[12:15]

    # Navigation equations. IMU State Estimation
    roll, pitch, yaw = rpy
    
    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    # Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    # Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    # Rbn = Rz @ Ry @ Rx
    Rbn = Rz
    accENU = Rbn@acc
        
    pIMU = pIMU + Ts*vIMU + Ts**2/2*(accENU+g) # position ENU
    vIMU = vIMU + Ts*(accENU+g) # velocity ENU
    W = np.array([
        [1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]
    ])
    rpy = rpy + Ts*W@gyr # attitude update
    
    # Wrap yaw to [-π, π]
    if rpy[2] < -np.pi:
        rpy[2] = 2*np.pi + rpy[2]
    if rpy[2] >= np.pi:
        rpy[2] = -2*np.pi + rpy[2]
    

    # Prediction
    fE, fN, fU = accENU  # specific force in navigation frame
    llaIMU = pm.enu2geodetic(pIMU[0],pIMU[1],pIMU[2],lla0[0],lla0[1],lla0[2])
    lat = radians(llaIMU[0])
    alt = llaIMU[2]
    
    M = a*(1-e2)/((1-e2*sin(lat)**2)**(3/2)) # meridian radius of curvature
    N = a/sqrt(1-e2*sin(lat)**2) # prime vertical radius of curvature

    # Dynamic matrix F (equation 6)
    F = np.zeros((15,15))
    # F_pv block: position to velocity coupling
    F[0:3,3:6] = np.eye(3)
    # -F_vε block: negative of skew-symmetric matrix -[f^n ×]
    F[3,7] = fU
    F[3,8] = -fN
    F[4,6] = -fU
    F[4,8] = fE
    F[5,6] = fN
    F[5,7] = -fE
    # Accelerometer bias coupling
    F[3:6,9:12] = -Rbn
    # Gyroscope bias coupling
    F[6:9,12:15] = -Rbn
    # Gauss-Markov bias dynamics
    F[9:12,9:12] = beta_acc*np.eye(3)
    F[12:15,12:15] = beta_gyr*np.eye(3)
    
    F = np.eye(15) + F*Ts
    P = F@P@F.T + Q
    x = F@x
    
    # Update
    if ((i+1)<A or (i+1)>B): # if out of GNSS outage simulation
    
        if not((i+1)%(frecIMU/frecGPS)): # if GNSS measurement is available
            
            K = P@H.T@np.linalg.inv(H@P@H.T + R)
            z = pm.geodetic2enu(lla[i,0],lla[i,1],lla[i,2],lla0[0],lla0[1],lla0[2]) - pIMU
            x = x + K@(z - H@x)
            P = (np.eye(15) - K@H)@P
    
            # Inject error state into nominal state
            pIMU += x[0:3]  # position correction
            vIMU += x[3:6]  # velocity correction
            # Orientation correction: x[6:9] = [δφ, δθ, δψ] = [δroll, δpitch, δyaw], rpy = [roll, pitch, yaw]
            rpy[0] += x[6]  # roll
            rpy[1] += x[7]  # pitch
            rpy[2] += x[8]  # yaw
        
    # Store current iteration results
    p[i+1,:] = pIMU.T  # position
    v[i+1,:] = vIMU.T  # velocity
    r[i+1,0] = rpy[0]  # roll (orient is [roll, pitch, yaw])
    r[i+1,1] = rpy[1]  # pitch
    r[i+1,2] = rpy[2]  # yaw
    
    # Error state reset after update (Error-State EKF)
    # After injecting errors into nominal state, reset error state to zero
    # Ref: Joan Solà 'Quaternion kinematics for the error-state KF'
    if ((i+1)<A or (i+1)>B):
        if not((i+1)%(frecIMU/frecGPS)):
            # 1. Get orientation error before reset (needed for covariance update)
            d_theta = x[6:9]  # [δroll, δpitch, δyaw]
            
            # 2. Compute reset Jacobian for small angle approximation
            # G_theta = I - [0.5*δθ]ₓ (first-order approximation)
            half_skew = np.array([
                [0, -0.5*d_theta[2], 0.5*d_theta[1]],
                [0.5*d_theta[2], 0, -0.5*d_theta[0]],
                [-0.5*d_theta[1], 0.5*d_theta[0], 0]
            ])
            G_theta = np.eye(3) - half_skew
            
            # 3. Update covariance for orientation block only (computational efficiency)
            P[6:9, :] = G_theta @ P[6:9, :]      # Update rows
            P[:, 6:9] = P[:, 6:9] @ G_theta.T    # Update columns
            
            # 4. Reset navigation error state to zero (pos, vel, orient)
            x[0:9] = 0

########################### RESULTS AND VISUALISATION ###########################

# PLOT THE TRAJECTORY
f = pm.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],lla0[0],lla0[1],lla0[2]) # convert the real trajectory from geodetic to ENU

# Create interactive figure with zoom capability
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(f[0],f[1],'k',linewidth=1.5, label='Ground Truth') # real trajectory in black
ax.plot(p[:,0],p[:,1],'b',linewidth=1.5, alpha=0.7, label='EKF Estimate') # estimated trajectory by the system in blue

if A!=0 or B!=0: # GNSS outage segment
    ax.plot(p[A:B,0],p[A:B,1],'r',linewidth=2, label='EKF during GPS outage') # system output in red
    ax.plot(f[0][A:B],f[1][A:B],'g',linewidth=2, label='Ground Truth during GPS outage') # real trajectory in green

ax.set_xlabel('East (m)', fontsize=12)
ax.set_ylabel('North (m)', fontsize=12)
ax.set_title('EKF Trajectory Estimation (Scroll to Zoom)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axis('equal')

# Enable interactive zooming with scroll wheel
plt.tight_layout()
plt.show()

########################### TRAJECTORY EVALUATION ###########################

# Convert ground truth trajectory to ENU array format
f = pm.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],lla0[0],lla0[1],lla0[2])
f_array = np.column_stack([f[0], f[1], f[2]])  # Nx3 array [E, N, U]

# Prepare GNSS outage information
gnss_outage_info = {
    'start': t1,
    'end': t2,
    'duration': d,
    'start_idx': A,
    'end_idx': B
}

# Run comprehensive evaluation
results = metrics.evaluate_navigation_performance(
    p_est=p,
    v_est=v,
    r_est=r,
    p_gt=f_array,
    v_gt=vel_enu,
    r_gt=orient,
    dataset_name=nav_data.dataset_name,
    gnss_outage_info=gnss_outage_info,
    sample_rate=frecIMU
)

# Create logs directory and save results
base_dir = Path(__file__).parent
logs_dir = os.path.join(base_dir, '../../../logs')
os.makedirs(logs_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'ekf_errors_{timestamp}.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log all evaluation results
metrics.log_evaluation_results(logger, results, log_file)








