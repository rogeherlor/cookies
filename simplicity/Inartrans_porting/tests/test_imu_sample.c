// Compile and run this test with the following command from the project root directory:

// gcc \
//   simplicity/Inartrans_porting/tests/test_imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -o simplicity/Inartrans_porting/tests/test_imu_sample

#include <stdio.h>
#include <stdbool.h>

#include "../src/sensors/imu_sample.h"

int main(void)
{
    CookieImuSample sample;

    CookieIMU_ClearSample(&sample);

    if (sample.valid) {
        printf("IMU clear test failed\n");
        return 1;
    }

    int32_t accel_mg[3] = {100, -20, 980};
    int32_t gyro_dps[3] = {1, 2, -3};
    uint32_t timestamp_ms = 12345;

    CookieIMU_SetSample(&sample, accel_mg, gyro_dps, timestamp_ms);

    if (!sample.valid) {
        printf("IMU sample is not valid\n");
        return 1;
    }

    printf("IMU sample created successfully\n");
    printf("Timestamp: %lu ms\n", (unsigned long)sample.timestamp_ms);
    printf("Accel: %ld, %ld, %ld mg\n",
           (long)sample.accel_mg[0],
           (long)sample.accel_mg[1],
           (long)sample.accel_mg[2]);
    printf("Gyro: %ld, %ld, %ld dps\n",
           (long)sample.gyro_dps[0],
           (long)sample.gyro_dps[1],
           (long)sample.gyro_dps[2]);

    return 0;
}
