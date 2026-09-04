#ifndef COOKIE_IMU_PREPROCESSOR_H
#define COOKIE_IMU_PREPROCESSOR_H

#include <stdbool.h>

#include "imu_converter.h"

typedef struct {
    bool valid;

    float accel_m_s2[3];
    float gyro_rad_s[3];
    float dt_s;
} CookieImuNavigationInput;

bool CookieIMU_PreprocessForNavigation(const CookieImuConvertedSample *input,
                                        CookieImuNavigationInput *output);

#endif
