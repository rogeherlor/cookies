#ifndef COOKIE_IMU_SAMPLE_H
#define COOKIE_IMU_SAMPLE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    bool valid;

    int32_t accel_mg[3];      // Acceleration in milli-g
    int32_t gyro_dps[3];      // Angular velocity in degrees per second
    uint32_t timestamp_ms;    // Sample timestamp in milliseconds
} CookieImuSample;

void CookieIMU_ClearSample(CookieImuSample *sample);

void CookieIMU_SetSample(CookieImuSample *sample,
                         const int32_t accel_mg[3],
                         const int32_t gyro_dps[3],
                         uint32_t timestamp_ms);

#endif
