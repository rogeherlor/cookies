#include "cookie_runtime.h"

#include <stdint.h>
#include <string.h>

#include "app_log.h"
#include "sl_sleeptimer.h"

#include "../app/app.h"
#include "../../platform/imu_icm20648/cookie_imu_platform.h"

/*
 * Real IMU acquisition period.
 *
 * The legacy project sampled the IMU at 200 Hz using a 5 ms periodic timer.
 * In this refactor we keep the same logical rate, but the scheduling is kept
 * inside the runtime layer instead of spreading timing logic across callbacks.
 */
#define COOKIE_RUNTIME_IMU_PERIOD_MS      5u

/*
 * Debug logging decimation.
 *
 * The IMU is read at 200 Hz, but printing every sample would flood the serial
 * console and could disturb the timing. Printing every 200 samples gives one
 * readable line per second while keeping the real acquisition path active.
 */
#define COOKIE_RUNTIME_IMU_LOG_EVERY_N    200u

/*
 * Single application context for the embedded runtime.
 *
 * This context belongs to the portable CookieApp layer. The runtime only feeds
 * it with hardware samples and later asks it to build messages.
 */
static CookieAppContext runtime_app;
static bool runtime_initialized = false;

static uint32_t last_imu_read_ms = 0u;
static uint32_t imu_sample_count = 0u;

/*
 * Convert sleeptimer ticks to milliseconds.
 *
 * Keeping this helper here avoids leaking Silicon Labs timing details into the
 * portable application logic.
 */
static uint32_t runtime_now_ms(void)
{
  /*
   * Some Gecko SDK versions expose a dedicated tick-count typedef and others do
   * not. In this project, sl_sleeptimer_get_tick_count() can be stored safely in
   * a uint32_t.
   */
  uint32_t ticks = sl_sleeptimer_get_tick_count();
  return sl_sleeptimer_tick_to_ms(ticks);
}

/*
 * Return true when enough time has elapsed to run a periodic task.
 *
 * Unsigned subtraction keeps the comparison valid even if the millisecond
 * counter wraps around.
 */
static bool runtime_period_elapsed(uint32_t now_ms,
                                   uint32_t last_ms,
                                   uint32_t period_ms)
{
  return (uint32_t)(now_ms - last_ms) >= period_ms;
}

/*
 * Read one real IMU sample and push it into the portable CookieApp layer.
 *
 * This function is intentionally small:
 *   - hardware access stays in CookiePlatformImu
 *   - conversion/preprocessing/navigation stay in CookieApp
 *   - this runtime only connects both worlds
 */
static void runtime_process_imu(uint32_t now_ms)
{
  CookiePlatformImuSample platform_sample = {0};

  if (!CookiePlatformImu_ReadSample(&platform_sample)) {
    app_log_info("CookieRuntime: IMU read failed\n");
    return;
  }

  /*
   * Use the runtime timestamp for now.
   *
   * Later, when GNSS time is integrated, this timestamp can be replaced or
   * corrected using the GPS-synchronised timestamp logic from the old project.
   */
  platform_sample.timestamp_ms = now_ms;

  CookieAppImuProcessDebug imu_debug = {0};

  bool processed = CookieApp_ProcessImuSampleWithDebug(&runtime_app,
                                                       platform_sample.accel_mg,
                                                       platform_sample.gyro_dps,
                                                       platform_sample.timestamp_ms,
                                                       &imu_debug);

  if (!processed) {
    app_log_info("CookieRuntime: IMU sample rejected by app layer\n");
    return;
  }

  imu_sample_count++;

  /*
   * Keep the serial output readable while still sampling the IMU at 200 Hz.
   */
  if ((imu_sample_count % COOKIE_RUNTIME_IMU_LOG_EVERY_N) == 0u) {
      app_log_info("CookieRuntime: IMU #%lu t=%lu ms | raw A=%ld,%ld,%ld mg | raw G=%ld,%ld,%ld dps | dt=%.3f s | nav_ready=%u | predicted=%u\n",
                   (unsigned long)imu_sample_count,
                   (unsigned long)platform_sample.timestamp_ms,
                   (long)platform_sample.accel_mg[0],
                   (long)platform_sample.accel_mg[1],
                   (long)platform_sample.accel_mg[2],
                   (long)platform_sample.gyro_dps[0],
                   (long)platform_sample.gyro_dps[1],
                   (long)platform_sample.gyro_dps[2],
                   (double)imu_debug.dt_s,
                   imu_debug.navigation_initialized ? 1u : 0u,
                   imu_debug.navigation_predicted ? 1u : 0u);

      app_log_info("CookieRuntime: converted A=%.2f,%.2f,%.2f m/s2 | G=%.3f,%.3f,%.3f rad/s\n",
                   (double)imu_debug.accel_m_s2[0],
                   (double)imu_debug.accel_m_s2[1],
                   (double)imu_debug.accel_m_s2[2],
                   (double)imu_debug.gyro_rad_s[0],
                   (double)imu_debug.gyro_rad_s[1],
                   (double)imu_debug.gyro_rad_s[2]);

      app_log_info("CookieRuntime: navigation input A=%.2f,%.2f,%.2f m/s2 | G=%.3f,%.3f,%.3f rad/s\n",
                   (double)imu_debug.navigation_accel_m_s2[0],
                   (double)imu_debug.navigation_accel_m_s2[1],
                   (double)imu_debug.navigation_accel_m_s2[2],
                   (double)imu_debug.navigation_gyro_rad_s[0],
                   (double)imu_debug.navigation_gyro_rad_s[1],
                   (double)imu_debug.navigation_gyro_rad_s[2]);
  }
}

bool CookieRuntime_Init(void)
{
  CookieAppConfig config;
  memset(&config, 0, sizeof(config));

  /*
   * Temporary default GNSS mode.
   *
   * This keeps the app context configured even though real GNSS is not wired
   * into the runtime yet.
   */
  config.gnss_mode = 7u;

  CookieApp_Init(&runtime_app, &config);

  app_log_info("CookieRuntime: init started\n");

  if (!CookiePlatformImu_Init()) {
    app_log_info("CookieRuntime: IMU init failed\n");
    runtime_initialized = false;
    return false;
  }

  last_imu_read_ms = runtime_now_ms();
  imu_sample_count = 0u;
  runtime_initialized = true;

  app_log_info("CookieRuntime: IMU ready, device ID = 0x%02X\n",
               CookiePlatformImu_GetDeviceId());

  app_log_info("CookieRuntime: IMU period = %lu ms, debug every %lu samples\n",
               (unsigned long)COOKIE_RUNTIME_IMU_PERIOD_MS,
               (unsigned long)COOKIE_RUNTIME_IMU_LOG_EVERY_N);

  app_log_info("CookieRuntime: init OK\n");

  return true;
}

void CookieRuntime_Process(void)
{
  if (!runtime_initialized) {
    return;
  }

  uint32_t now_ms = runtime_now_ms();

  if (runtime_period_elapsed(now_ms,
                             last_imu_read_ms,
                             COOKIE_RUNTIME_IMU_PERIOD_MS)) {
    /*
     * Advance by the nominal period instead of assigning now_ms directly.
     *
     * This avoids accumulating drift if CookieRuntime_Process() is called a bit
     * late once. If the system is heavily delayed, the runtime still processes
     * one sample per call and catches up progressively.
     */
    last_imu_read_ms += COOKIE_RUNTIME_IMU_PERIOD_MS;

    runtime_process_imu(now_ms);
  }
}
