#include "imu_converter.h"

#include <stdint.h>
#include <string.h>

#define COOKIE_GRAVITY_M_S2 9.81f
#define COOKIE_DEG_TO_RAD   0.017453292519943295f

static bool has_previous_timestamp = false;
static uint32_t previous_timestamp_ms = 0U;

void CookieIMU_ConverterReset(void)
{
    has_previous_timestamp = false;
    previous_timestamp_ms = 0U;
}

bool CookieIMU_ConvertSample(const CookieImuSample *sample,
                             CookieImuConvertedSample *converted)
{
    if (sample == NULL || converted == NULL || !sample->valid) {
        return false;
    }

    memset(converted, 0, sizeof(*converted));

    for (int i = 0; i < 3; i++) {
        converted->accel_m_s2[i] = ((float)sample->accel_mg[i] / 1000.0f) * COOKIE_GRAVITY_M_S2;
        converted->gyro_rad_s[i] = ((float)sample->gyro_dps[i]) * COOKIE_DEG_TO_RAD;
    }

    if (!has_previous_timestamp) {
        converted->dt_s = 0.0f;
        has_previous_timestamp = true;
    } else {
        uint32_t delta_ms = sample->timestamp_ms - previous_timestamp_ms;
        converted->dt_s = ((float)delta_ms) / 1000.0f;
    }

    previous_timestamp_ms = sample->timestamp_ms;
    converted->valid = true;

    return true;
}
