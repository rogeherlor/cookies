#include "cookie_imu_platform.h"

#include <stddef.h>

#include "app_log.h"

#include "icm20648_r.h"

/*
 * Local state kept inside the platform adapter.
 *
 * This avoids exposing legacy driver details to the rest of the application.
 */
static bool imu_initialised = false;
static uint8_t imu_device_id = 0U;

bool CookiePlatformImu_Init(void)
{
  uint32_t status;

  app_log_info("CookiePlatformImu: init started\n");

  /*
   * Initialise the legacy ICM20648/ICM20948 driver.
   *
   * Internally, this configures the SPI bus using USART2, enables the IMU,
   * disables the IMU I2C interface, and reads the WHO_AM_I register.
   */
  status = ICM20648_init();
  if (status != ICM20648_OK) {
    app_log_error("CookiePlatformImu: ICM20648_init failed, status=0x%08lX\n",
                  (unsigned long)status);
    imu_initialised = false;
    return false;
  }

  /*
   * Read the device ID for debug purposes.
   *
   * In our successful test, this returned 0xE0.
   */
  status = ICM20648_getDeviceID(&imu_device_id);
  if (status != ICM20648_OK) {
    app_log_error("CookiePlatformImu: getDeviceID failed, status=0x%08lX\n",
                  (unsigned long)status);
    imu_initialised = false;
    return false;
  }

  app_log_info("CookiePlatformImu: device ID = 0x%02X\n", imu_device_id);

  /*
   * Enable accelerometer and gyroscope.
   *
   * Third argument is temperature sensor enable. We keep it disabled for now
   * because the current pipeline only needs acceleration and angular rate.
   */
  ICM20648_sensorEnable(true, true, false);

  /*
   * Configure the same basic ranges used in the previous bring-up test.
   *
   * Accelerometer:
   *   +/- 2 g, then converted to mg in ReadSample().
   *
   * Gyroscope:
   *   +/- 250 dps, read directly in degrees per second.
   */
  ICM20648_accelFullscaleSet(ICM20648_ACCEL_FULLSCALE_2G);
  ICM20648_gyroFullscaleSet(ICM20648_GYRO_FULLSCALE_250DPS);

  /*
   * Use 24 Hz bandwidth, matching the corrected old INARTRANS IMU setup.
   *
   * This is a reasonable first value for navigation-like dynamics because it
   * reduces high-frequency noise while preserving the movement we care about.
   */
  ICM20648_accelBandwidthSet(ICM20648_ACCEL_BW_24HZ);
  ICM20648_gyroBandwidthSet(ICM20648_GYRO_BW_24HZ);

  imu_initialised = true;

  app_log_info("CookiePlatformImu: init OK\n");

  return true;
}

bool CookiePlatformImu_ReadSample(CookiePlatformImuSample *sample)
{
  float accel_g[3];
  float gyro_dps[3];

  if (sample == NULL) {
    return false;
  }

  sample->valid = false;

  if (!imu_initialised) {
    app_log_error("CookiePlatformImu: read requested before init\n");
    return false;
  }

  /*
   * Read raw physical values through the legacy driver.
   *
   * ICM20648_accelDataRead() returns acceleration in g.
   * ICM20648_gyroDataRead() returns angular rate in deg/s.
   */
  if (ICM20648_accelDataRead(accel_g) != ICM20648_OK) {
    app_log_error("CookiePlatformImu: accel read failed\n");
    return false;
  }

  if (ICM20648_gyroDataRead(gyro_dps) != ICM20648_OK) {
    app_log_error("CookiePlatformImu: gyro read failed\n");
    return false;
  }

  /*
   * Convert to the integer units used by the portable pipeline.
   *
   * Acceleration:
   *   g -> mg
   *
   * Gyroscope:
   *   deg/s -> integer deg/s for now.
   *
   * Later, if the navigation pipeline needs finer gyro resolution, we can
   * switch to centi-dps or milli-dps, but for now we keep the same style as
   * the old debug output.
   */
  for (uint8_t i = 0; i < 3; i++) {
    sample->accel_mg[i] = (int32_t)(accel_g[i] * 1000.0f);
    sample->gyro_dps[i] = (int32_t)(gyro_dps[i]);
  }

  /*
   * Timestamp is intentionally left as 0 for this first adapter step.
   *
   * The next step will decide whether the timestamp comes from:
   *   - sleeptimer ticks,
   *   - GNSS-synchronised time,
   *   - or the portable app orchestration layer.
   */
  sample->timestamp_ms = 0U;
  sample->valid = true;

  return true;
}

uint8_t CookiePlatformImu_GetDeviceId(void)
{
  return imu_device_id;
}
