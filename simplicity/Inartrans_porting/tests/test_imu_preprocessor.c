#include <stdio.h>

#include "../src/sensors/imu_sample.h"
#include "../src/sensors/imu_converter.h"
#include "../src/sensors/imu_preprocessor.h"

// gcc \
//   simplicity/Inartrans_porting/tests/test_imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -o simplicity/Inartrans_porting/tests/test_imu_preprocessor

// ./simplicity/Inartrans_porting/tests/test_imu_preprocessor

int main(void)
{
    CookieIMU_ConverterReset();

    CookieImuSample raw;
    int32_t accel_mg[3] = {1000, 2000, -3000};
    int32_t gyro_dps[3] = {10, -20, 30};

    CookieIMU_SetSample(&raw, accel_mg, gyro_dps, 1000);

    CookieImuConvertedSample converted;

    if (!CookieIMU_ConvertSample(&raw, &converted)) {
        printf("IMU conversion failed\n");
        return 1;
    }

    CookieImuNavigationInput nav_input;

    if (!CookieIMU_PreprocessForNavigation(&converted, &nav_input)) {
        printf("IMU preprocessing failed\n");
        return 1;
    }

    printf("IMU preprocessing successful\n");

    printf("Converted accel: %.3f, %.3f, %.3f m/s^2\n",
           converted.accel_m_s2[0],
           converted.accel_m_s2[1],
           converted.accel_m_s2[2]);

    printf("Navigation accel: %.3f, %.3f, %.3f m/s^2\n",
           nav_input.accel_m_s2[0],
           nav_input.accel_m_s2[1],
           nav_input.accel_m_s2[2]);

    printf("Converted gyro: %.6f, %.6f, %.6f rad/s\n",
           converted.gyro_rad_s[0],
           converted.gyro_rad_s[1],
           converted.gyro_rad_s[2]);

    printf("Navigation gyro: %.6f, %.6f, %.6f rad/s\n",
           nav_input.gyro_rad_s[0],
           nav_input.gyro_rad_s[1],
           nav_input.gyro_rad_s[2]);

    printf("dt: %.3f s\n", nav_input.dt_s);

    return 0;
}
