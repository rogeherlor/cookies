# Objectives:

Kinematics. Solà’s ESKF Equations. Physics is already "solved"; no need to re-learn gravity or basic integration.

Noise Modeling ($Q, R$). Neural Network (CNN/LSTM). Networks are great at identifying "vibration" or "motion patterns" to tune uncertainty dynamically.

Bias Correction. Neural Network. Deep learning can model complex, non-linear sensor drifts better than a constant-bias assumption.


# Pending SOTA and possible implementations:
## ESKF:

TLIO (Tight-Learned Inertial Odometry): Uses a ResNet to regress displacement and uncertainty, which is then fed into an EKF. It essentially uses the network as a "smart" measurement provider for an ESKF.

Differentiable Kalman Filters (DKF): These treat the entire ESKF as a layer. Because Solà’s equations are differentiable, you can backpropagate through the entire filter to train the network to produce the best possible state estimate.

AI-IMU Dead-Reckoning (2020)

ES-KalmanNet (2025/Recent)

RBF-ESKF (2021)

## IEKF:

AI-IMU Dead-Reckoning (Brossard et al.): This is a landmark paper that uses a Neural Network to estimate the noise parameters of an Invariant EKF.

# Pending Improve Cookies Sensor Data

# Visualize IMU alone