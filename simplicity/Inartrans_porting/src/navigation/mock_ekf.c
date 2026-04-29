#include "mock_ekf.h"

void EKF_Init(EKF_Context_t *ekf, float lat0, float lon0, float alt0)
{
    ekf->lla0[0] = lat0;
    ekf->lla0[1] = lon0;
    ekf->lla0[2] = alt0;

    ekf->pos_enu[0] = 0.0f;
    ekf->pos_enu[1] = 0.0f;
    ekf->pos_enu[2] = 0.0f;

    ekf->vel_enu[0] = 0.0f;
    ekf->vel_enu[1] = 0.0f;
    ekf->vel_enu[2] = 0.0f;
}

void EKF_Update(EKF_Context_t *ekf, float lat, float lon, float alt)
{
    // Fake: directly store LLA difference as ENU
    ekf->pos_enu[0] = lat - ekf->lla0[0];
    ekf->pos_enu[1] = lon - ekf->lla0[1];
    ekf->pos_enu[2] = alt - ekf->lla0[2];
}

void EKF_Predict(EKF_Context_t *ekf, float *acc, float *gyr, float dt)
{
    // Fake: integrate accel into velocity
    for (int i = 0; i < 3; i++) {
        ekf->vel_enu[i] += acc[i] * dt;
    }
}

void ENU_to_LLA(float *enu, float lat0, float lon0, float alt0, float *lla)
{
    lla[0] = lat0 + enu[0];
    lla[1] = lon0 + enu[1];
    lla[2] = alt0 + enu[2];
}