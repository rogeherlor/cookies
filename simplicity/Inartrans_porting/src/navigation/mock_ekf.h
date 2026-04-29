#ifndef MOCK_EKF_H
#define MOCK_EKF_H

#include <stdint.h>

typedef struct {
    float pos_enu[3];
    float vel_enu[3];
    float lla0[3];
} EKF_Context_t;

void EKF_Init(EKF_Context_t *ekf, float lat0, float lon0, float alt0);
void EKF_Update(EKF_Context_t *ekf, float lat, float lon, float alt);
void EKF_Predict(EKF_Context_t *ekf, float *acc, float *gyr, float dt);

void ENU_to_LLA(float *enu, float lat0, float lon0, float alt0, float *lla);

#endif