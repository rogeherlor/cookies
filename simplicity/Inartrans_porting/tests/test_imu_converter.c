#include <stdio.h>

#include "../src/sensors/imu_sample.h"
#include "../src/sensors/imu_converter.h"

// gcc \
//   simplicity/Inartrans_porting/tests/test_imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -o simplicity/Inartrans_porting/tests/test_imu_converter

// ./simplicity/Inartrans_porting/tests/test_imu_converter

int main(void)
{
    CookieIMU_ConverterReset();

    CookieImuSample sample_1;
    int32_t accel_1[3] = {1000, 0, -1000};
    int32_t gyro_1[3] = {0, 90, -180};

    CookieIMU_SetSample(&sample_1, accel_1, gyro_1, 1000);

    CookieImuConvertedSample converted_1;

    if (!CookieIMU_ConvertSample(&sample_1, &converted_1)) {
        printf("First IMU conversion failed\n");
        return 1;
    }

    printf("First sample\n");
    printf("Accel: %.3f, %.3f, %.3f m/s^2\n",
           converted_1.accel_m_s2[0],
           converted_1.accel_m_s2[1],
           converted_1.accel_m_s2[2]);
    printf("Gyro: %.6f, %.6f, %.6f rad/s\n",
           converted_1.gyro_rad_s[0],
           converted_1.gyro_rad_s[1],
           converted_1.gyro_rad_s[2]);
    printf("dt: %.3f s\n", converted_1.dt_s);

    CookieImuSample sample_2;
    int32_t accel_2[3] = {0, 0, 1000};
    int32_t gyro_2[3] = {10, 20, 30};

    CookieIMU_SetSample(&sample_2, accel_2, gyro_2, 1010);

    CookieImuConvertedSample converted_2;

    if (!CookieIMU_ConvertSample(&sample_2, &converted_2)) {
        printf("Second IMU conversion failed\n");
        return 1;
    }

    printf("Second sample\n");
    printf("Accel: %.3f, %.3f, %.3f m/s^2\n",
           converted_2.accel_m_s2[0],
           converted_2.accel_m_s2[1],
           converted_2.accel_m_s2[2]);
    printf("Gyro: %.6f, %.6f, %.6f rad/s\n",
           converted_2.gyro_rad_s[0],
           converted_2.gyro_rad_s[1],
           converted_2.gyro_rad_s[2]);
    printf("dt: %.3f s\n", converted_2.dt_s);

    return 0;
}
