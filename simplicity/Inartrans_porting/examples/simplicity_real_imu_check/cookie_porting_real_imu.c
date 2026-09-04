#include "cookie_porting_real_imu.h"

#include "app_log.h"

#include "porting/platform/imu_icm20648/cookie_imu_platform.h"

/*
 * One-shot real IMU bring-up test.
 *
 * This is still a temporary debug function, but now it uses the clean platform
 * adapter instead of calling the legacy ICM20648 driver directly.
 */
void CookiePorting_RunRealImuCheck(void)
{
  app_log_info("Real IMU check started\n");

  if (!CookiePlatformImu_Init()) {
    app_log_error("Real IMU check failed: init error\n");
    return;
  }

  app_log_info("IMU device ID: 0x%02X\n", CookiePlatformImu_GetDeviceId());

  for (uint8_t i = 0; i < 5; i++) {
    CookiePlatformImuSample sample;

    if (!CookiePlatformImu_ReadSample(&sample)) {
      app_log_error("IMU sample %u failed\n", (unsigned int)(i + 1U));
      continue;
    }

    app_log_info("IMU sample %u: A=%ld,%ld,%ld mg | G=%ld,%ld,%ld dps\n",
                 (unsigned int)(i + 1U),
                 (long)sample.accel_mg[0],
                 (long)sample.accel_mg[1],
                 (long)sample.accel_mg[2],
                 (long)sample.gyro_dps[0],
                 (long)sample.gyro_dps[1],
                 (long)sample.gyro_dps[2]);
  }

  app_log_info("Real IMU check finished\n");
}
