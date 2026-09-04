#ifndef COOKIE_EKF_H_
#define COOKIE_EKF_H_

#include "CMSIS_5/CMSIS/DSP/Include/arm_math.h"
#include <math.h>
#include <stdbool.h>

// State vector size
#define EKF_STATE_DIM 15
#define EKF_MEAS_DIM 3

typedef struct {
    // Navigation State
    float pos_enu[3];    // Position (m) [East, North, Up]
    float vel_enu[3];    // Velocity (m/s) [East, North, Up]
    float orient_ypr[3]; // Orientation (rad) [Yaw, Pitch, Roll]

    // Reference LLA for ENU conversion (Lat, Lon, Alt)
    float lla0[3]; 
    
    // Kalman Filter Matrices (CMSIS-DSP instances)
    arm_matrix_instance_f32 P; // Covariance (15x15)
    arm_matrix_instance_f32 Q; // Process Noise (15x15)
    arm_matrix_instance_f32 R; // Measurement Noise (3x3)
    arm_matrix_instance_f32 x; // Error State Vector (15x1)
    
    // Internal buffers for matrix operations (to avoid stack overflow)
    float P_data[EKF_STATE_DIM * EKF_STATE_DIM];
    float Q_data[EKF_STATE_DIM * EKF_STATE_DIM];
    float R_data[EKF_MEAS_DIM * EKF_MEAS_DIM];
    float x_data[EKF_STATE_DIM];
    
    // Biases (Part of state)
    float acc_bias[3];
    float gyr_bias[3];
    
} EKF_Context_t;

// Initialization
void EKF_Init(EKF_Context_t *ekf, float lat0, float lon0, float alt0);

// Prediction Step (IMU @ 10Hz)
// acc: [ax, ay, az] in m/s^2
// gyr: [gx, gy, gz] in rad/s
// dt: time step in seconds
void EKF_Predict(EKF_Context_t *ekf, float *acc, float *gyr, float dt);

// Update Step (GPS @ 1Hz)
// lat, lon: degrees
// alt: meters
void EKF_Update(EKF_Context_t *ekf, float lat, float lon, float alt);

// Coordinate Utilities
void LLA_to_ENU(float lat, float lon, float alt, float lat0, float lon0, float alt0, float *enu);
void ENU_to_LLA(float *enu, float lat0, float lon0, float alt0, float *lla);

#endif /* COOKIE_EKF_H_ */
