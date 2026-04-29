#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "../src/gnss/gnss.h"
#include "../src/gnss/gnss_converter.h"

#include "../src/sensors/imu_sample.h"
#include "../src/sensors/imu_converter.h"
#include "../src/sensors/imu_preprocessor.h"

#include "../src/navigation/navigation.h"

// cookie_ekf
// gcc \
//   simplicity/Inartrans_porting/tests/test_application.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   simplicity/Inartrans_porting/src/gnss/gnss_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/navigation/navigation.c \
//   simplicity/Inartrans_v2/ekf/cookie_ekf.c \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -I simplicity/Inartrans_porting/src/navigation \
//   -I simplicity/Inartrans_v2/ekf \
//   -lm \
//   -o simplicity/Inartrans_porting/tests/test_application

// mock
// gcc \
//   simplicity/Inartrans_porting/tests/test_application.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   simplicity/Inartrans_porting/src/gnss/gnss_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/navigation/navigation.c \
//   simplicity/Inartrans_porting/src/navigation/mock_ekf.c \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -I simplicity/Inartrans_porting/src/navigation \
//   -lm \
//   -o simplicity/Inartrans_porting/tests/test_application


#define IMU_FREQ_HZ 100
#define GNSS_FREQ_HZ 1
#define SIM_TIME_SEC 12

int main(void)
{
    printf("Starting navigation test...\n");

    CookieNavigation_Init();
    CookieIMU_ConverterReset();

    uint32_t timestamp_ms = 0;

    for (int step = 0; step < SIM_TIME_SEC * IMU_FREQ_HZ; step++)
    {
        // ---------------------------
        // 1. IMU (fake data)
        // ---------------------------
        CookieImuSample imu_raw;

        int32_t accel_mg[3] = {0, 0, 0};   // static, gravity only
        int32_t gyro_dps[3] = {0, 0, 0};      // no rotation

        CookieIMU_SetSample(&imu_raw, accel_mg, gyro_dps, timestamp_ms);

        CookieImuConvertedSample imu_conv;
        CookieImuNavigationInput imu_nav;

        if (CookieIMU_ConvertSample(&imu_raw, &imu_conv))
        {
            if (CookieIMU_PreprocessForNavigation(&imu_conv, &imu_nav))
            {
                CookieNavigation_PredictWithImu(&imu_nav);
            }
        }

        // ---------------------------
        // 2. GNSS (1 Hz)
        // ---------------------------
        if (step % IMU_FREQ_HZ == 0)
        {
            CookieGnssFix fix;
            memset(&fix, 0, sizeof(fix));

            fix.valid = true;

            fix.latitude_raw = 4807.038f;
            fix.latitude_dir = 'N';

            fix.longitude_raw = 1131.000f;
            fix.longitude_dir = 'E';

            fix.altitude_m = 545.4f;

            CookieNavigation_UpdateWithGnss(&fix);

            printf("[GNSS] Update\n");
        }

        // ---------------------------
        // 3. Estado
        // ---------------------------
        CookieNavigationState state;

        if (CookieNavigation_GetState(&state))
        {
            printf("State: lat=%.6f lon=%.6f alt=%.2f vel=%.2f\n",
                   state.latitude_deg,
                   state.longitude_deg,
                   state.altitude_m,
                   state.velocity_m_s);
        }

        timestamp_ms += 10; // 100 Hz → 10 ms
    }

    printf("Test finished.\n");
    return 0;
}
