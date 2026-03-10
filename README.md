# Objectives:

Kinematics. Solà’s ESKF Equations. Physics is already "solved"; no need to re-learn gravity or basic integration.

Noise Modeling ($Q, R$). Neural Network (CNN/LSTM). Networks are great at identifying "vibration" or "motion patterns" to tune uncertainty dynamically.

Bias Correction. Neural Network. Deep learning can model complex, non-linear sensor drifts better than a constant-bias assumption.


# Pending SOTA and possible implementations:
## ESKF:

TLIO (Tight-Learned Inertial Odometry): Uses a ResNet to regress displacement and uncertainty, which is then fed into an EKF. It essentially uses the network as a "smart" measurement provider for an ESKF.

Differentiable Kalman Filters (DKF): These treat the entire ESKF as a layer. Because Solà’s equations are differentiable, you can backpropagate through the entire filter to train the network to produce the best possible state estimate.

ES-KalmanNet (2025/Recent)

RBF-ESKF (2021)

## IEKF:

AI-IMU Dead-Reckoning (Brossard et al.): This is a landmark paper that uses a Neural Network to estimate the noise parameters of an Invariant EKF.

## Full DL:

Herath et al. (RoNIN)

Zhao et al., "Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics" (CVPR 2025)

## Smoother

Use it as ground truth instead of the actual one

# Pending Improve Cookies Sensor Data

# Pending benchmark metrics
Maybe the same as in AI-IMU Dead-Reckoning, KITTI scenaro.