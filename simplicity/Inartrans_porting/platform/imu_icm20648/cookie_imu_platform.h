#ifndef COOKIE_IMU_PLATFORM_H
#define COOKIE_IMU_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Clean platform-facing IMU sample.
 *
 * This structure is intentionally independent from the ICM20648 driver.
 * The rest of the application should use this type instead of calling
 * the legacy IMU driver directly.
 */
typedef struct {
  int32_t accel_mg[3];      /* Acceleration in milli-g. 1000 mg ~= 1 g. */
  int32_t gyro_dps[3];      /* Angular rate in degrees per second. */
  uint32_t timestamp_ms;    /* Timestamp in milliseconds. Currently filled by caller/platform. */
  bool valid;               /* True if the sample was read successfully. */
} CookiePlatformImuSample;

/*
 * Initialise the real IMU hardware.
 *
 * Returns true if the IMU was initialised correctly.
 */
bool CookiePlatformImu_Init(void);

/*
 * Read one accelerometer + gyroscope sample from the real IMU.
 *
 * Returns true if the sample was read correctly.
 */
bool CookiePlatformImu_ReadSample(CookiePlatformImuSample *sample);

/*
 * Return the last IMU device ID read during initialisation.
 */
uint8_t CookiePlatformImu_GetDeviceId(void);

#endif /* COOKIE_IMU_PLATFORM_H */
