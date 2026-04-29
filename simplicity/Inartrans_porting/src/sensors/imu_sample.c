#include "imu_sample.h"

#include <string.h>

void CookieIMU_ClearSample(CookieImuSample *sample)
{
    if (sample == NULL) {
        return;
    }

    memset(sample, 0, sizeof(*sample));
    sample->valid = false;
}

void CookieIMU_SetSample(CookieImuSample *sample,
                         const int32_t accel_mg[3],
                         const int32_t gyro_dps[3],
                         uint32_t timestamp_ms)
{
    if (sample == NULL || accel_mg == NULL || gyro_dps == NULL) {
        return;
    }

    sample->valid = true;
    sample->timestamp_ms = timestamp_ms;

    for (int i = 0; i < 3; i++) {
        sample->accel_mg[i] = accel_mg[i];
        sample->gyro_dps[i] = gyro_dps[i];
    }
}
