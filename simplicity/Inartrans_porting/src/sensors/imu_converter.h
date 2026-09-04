#ifndef COOKIE_IMU_CONVERTER_H
#define COOKIE_IMU_CONVERTER_H

#include <stdbool.h>

#include "imu_sample.h"

typedef struct {
    bool valid;

    float accel_m_s2[3];   // Acceleration in m/s^2
    float gyro_rad_s[3];   // Angular velocity in rad/s
    float dt_s;            // Time difference from previous sample in seconds
} CookieImuConvertedSample;

void CookieIMU_ConverterReset(void);

bool CookieIMU_ConvertSample(const CookieImuSample *sample,
                             CookieImuConvertedSample *converted);

#endif
