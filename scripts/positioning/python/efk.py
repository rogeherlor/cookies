import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from math import sqrt
from math import sin
from math import cos
from math import tan
from math import radians
import pymap3d as pm
from scipy.io import loadmat
from pathlib import Path
from sklearn.metrics import mean_squared_error

################### LOAD DATA ###########################
# Load raw KITTI ('10_03_0034.mat')
base_dir = Path(__file__).parent
data_path = os.path.join(base_dir, '../../../', 'datasets','raw_kitti', '10_03_0034.mat')
data = loadmat(data_path)

accel_flu = data['accel_flu']
gyro_flu = data['gyro_flu']
vel_enu = data['vel_enu']
lla = data['lla']
orient = data['rpy']

accel_flu = pd.DataFrame(accel_flu)
accel_flu.columns = ['accelX','accelY','accelZ']
accel_flu.to_csv('accel_flu.csv')
accel_flu = read_csv('accel_flu.csv', header=0, index_col=0)
accel_flu = accel_flu.values

gyro_flu = pd.DataFrame(gyro_flu)
gyro_flu.columns = ['gyroX','gyroY','gyroZ']
gyro_flu.to_csv('gyro_flu.csv')
gyro_flu = read_csv('gyro_flu.csv', header=0, index_col=0)
gyro_flu = gyro_flu.values

vel_enu = pd.DataFrame(vel_enu)
vel_enu.columns = ['velE','velN','velU']
vel_enu.to_csv('vel_enu.csv')
vel_enu = read_csv('vel_enu.csv', header=0, index_col=0)
vel_enu = vel_enu.values

lla = pd.DataFrame(lla)
lla.columns = ['lat','lon','alt']
lla.to_csv('lla.csv')
lla = read_csv('lla.csv', header=0, index_col=0)
lla = lla.values

orient = pd.DataFrame(orient)
orient.columns = ['roll','pitch','yaw']
orient.to_csv('orient.csv')
orient = read_csv('orient.csv', header=0, index_col=0)
orient = orient.values

########################## VAR INIT ###############
# No GNSS loss: t1=0, d=0
t1 = 0  # time init loss in seconds
d = 0   # duration of GNSS loss in seconds
frecIMU = 10        # 10 Hz
frecGPS = 1         # 1 Hz

lla0 = np.array([49.01,8.43,116.43])    # origin of ENU coordinate system
g = np.array([0,0,9.81])                # gravity acceleration
a = 6378137.0                           # equatorial radius [m]
b = 6356752.3142                        # polar radius [m]
e2 = 0.00669437999                      # eccentricity^2 (1-b^2/a^2)
ep2 = 0.00673949674                     # second eccentricity^2 (a^2/b^2 -1)

t2 = t1+d                               # time end loss in seconds
A = t1*frecIMU                          # data index start loss
B = t2*frecIMU                          # data index end loss
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
ypr = np.zeros(3)   # unkwon attitude (yaw, pitch, roll)

## Filter Initialisation
beta_acc = 3.7e-7 # Gauss-Markov coef # TODO: get from modelling
beta_gyr = 2.9e-1 # Gauss-Markov coef # TODO: get from modelling

# State initialisation x
accBias = 1e-6      # bias inicial
gyrBias = 1e-7      # bias inicial
x = np.zeros(15)    # state x = (δp[3], δv[3], δϵ[3], b_acc[3], b_gyr[3])
                    # δϵ = [δroll, δpitch, δyaw] (indices 6,7,8)
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
P = np.eye(15)*0.1 # TODO: get from modelling
K = np.zeros((15,3))

################## RUN #########################
for i in range(0,NN-1):
    
    # IMU Correction
    acc = accel_flu[i,:] + x[9:12]
    gyr = gyro_flu[i,:] + x[12:15]

    # Navigation equations. IMU State Estimation
    yaw, pitch, roll = ypr
    
    Rbn = np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])
    accENU = Rbn@acc
        
    pIMU = pIMU + Ts*vIMU + Ts**2/2*(accENU-g) # position ENU
    vIMU = vIMU + Ts*(accENU-g) # velocity ENU
    W = np.array([
        [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)],
        [0, cos(roll), -sin(roll)],
        [1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)]
    ])
    ypr = ypr + Ts*W@gyr # attitude update
    
    # Wrap yaw to [-π, π]
    if ypr[0] < -np.pi:
        ypr[0] = 2*np.pi + ypr[0]
    if ypr[0] >= np.pi:
        ypr[0] = -2*np.pi + ypr[0]
    

    # Prediction
    accE, accN, accU = accENU
    llaIMU = pm.enu2geodetic(pIMU[0],pIMU[1],pIMU[2],lla0[0],lla0[1],lla0[2])
    lat = radians(llaIMU[0])
    alt = llaIMU[2]
    
    M = a*(1-e2)/((1-e2*sin(lat)**2)**(3/2)) # radio de curvatura en el meridiano
    N = a/sqrt(1-e2*sin(lat)**2) # radio de curvatura en el plano vertical principal

    # Dynamic matrix F
    F = np.zeros((15,15))
    F[0:3,3:6] = np.eye(3)
    F[3,7] = accU
    F[3,8] = -accN
    F[4,6] = -accU
    F[4,8] = accE
    F[5,6] = accN
    F[5,7] = -accE
    F[6,4] = 1/(M+alt)
    F[7,3] = 1/(N+alt)
    F[8,3] = tan(lat)/(N+alt)
    F[3:6,9:12] = Rbn
    F[6:9,12:15] = Rbn
    F[9:12,9:12] = -beta_acc*np.eye(3)
    F[12:15,12:15] = -beta_gyr*np.eye(3)
    
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
            # Orientation correction: x[6:9] = [δroll, δpitch, δyaw]
            ypr[2] += x[6]  # roll
            ypr[1] += x[7]  # pitch  
            ypr[0] += x[8]  # yaw
        
    # Store current iteration results
    p[i+1,:] = pIMU.T  # position
    v[i+1,:] = vIMU.T  # velocity
    r[i+1,0] = ypr[2]  # roll
    r[i+1,1] = ypr[1]  # pitch
    r[i+1,2] = ypr[0]  # yaw
    
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
#%%
########################### MOSTRAR RESULTADOS Y ERRORES ########################

# DIBUJAR LA TRAYECTORIA

f = pm.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],lla0[0],lla0[1],lla0[2]) # pasar la trayectoria real de geodesicas a ENU

plt.plot(f[0],f[1],'k',label='Trayectoria') # trayectoria real en negro
plt.plot(p[:,0],p[:,1],'b',label='EKF') # trayectoria estimada por el sistema en azul

if A!=0 or B!=0: # tramo perdida de datos de GPS 
    plt.plot(p[A:B,0],p[A:B,1],'r',label='EKF en pérdida GPS') # salida del sistema en rojo
    plt.plot(f[0][A:B],f[1][A:B],'g',label='Trayectoria en pérdida GPS' ) # trayectoria real en verde
plt.legend()
plt.show()

# ERRORES 

# Error posicion en coordenadas ENU
pRMS_E = sqrt(mean_squared_error(f[0],p[:,0]))
pRMS_N = sqrt(mean_squared_error(f[1],p[:,1]))
pRMS_U = sqrt(mean_squared_error(f[2],p[:,2]))
    
# Error velocidad en coordenadas ENU
vRMS_E = sqrt(mean_squared_error(vel_enu[:,0],v[:,0]))
vRMS_N = sqrt(mean_squared_error(vel_enu[:,1],v[:,1]))
vRMS_U = sqrt(mean_squared_error(vel_enu[:,2],v[:,2]))

# Error orientacion en angulos de Euler
rRMS_R = sqrt(mean_squared_error(orient[:,0],r[:,0]))
rRMS_P = sqrt(mean_squared_error(orient[:,1],r[:,1]))
rRMS_Y = sqrt(mean_squared_error(orient[:,2],r[:,2]))
    
print('Error en posicion E: %.3f' % pRMS_E)
print('N: %.3f' %pRMS_N)
print('U: %.3f' %pRMS_U)

print('Error en velocidad E: %.3f' % vRMS_E )
print('N: %.3f' %vRMS_N)
print('U: %.3f' %vRMS_U)

print('Error en orientacion roll: %.3f' % rRMS_R)
print('pitch: %.3f' %rRMS_P)
print('yaw: %.3f' %rRMS_Y)

# Errores absolutos de posicion 
error_E_EKF=np.zeros((p.shape[0],1))
error_N_EKF=np.zeros((p.shape[0],1))

for i in range(0,p.shape[0]-1):
    error_E_EKF[i]= abs(f[0][i]-p[i,0])
    error_N_EKF[i]= abs(f[1][i]-p[i,1])
        
# Errores absolutos de orientacion 
error_yaw_EKF=np.zeros((1000,1))

for i in range(0,1000):
    error_yaw_EKF[i]= abs(orient[i,2]-r[i,2])








