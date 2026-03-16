# Objectives:

Kinematics. Solà’s ESKF Equations. Physics is already "solved"; no need to re-learn gravity or basic integration.

Noise Modeling ($Q, R$). Neural Network (CNN/LSTM). Networks are great at identifying "vibration" or "motion patterns" to tune uncertainty dynamically.

Bias Correction. Neural Network. Deep learning can model complex, non-linear sensor drifts better than a constant-bias assumption.


# Pending SOTA and possible implementations:
## ES-EKF:

DKF Deep Kalman filter: Simultaneous multi-sensor integration and modelling; A GNSS/IMU case study

## EKF

TLIO (Tight-Learned Inertial Odometry): Uses a ResNet to regress displacement and uncertainty, which is then fed into an EKF. It essentially uses the network as a "smart" measurement provider for an ESKF.

## IEKF:

AI-IMU Dead-Reckoning (Brossard et al.): This is a landmark paper that uses a Neural Network to estimate the noise parameters of an Invariant EKF.

## Full DL:

Zhao et al., "Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics" (CVPR 2025)

# Pending Improve Cookies Sensor Data

# Pending benchmark metrics
Same as in AI-IMU Dead-Reckoning, KITTI scenaro.
Replicate their results and include ours.

# Check the only imu track equations

# Pending conversion from the sensor to the vehicle frame