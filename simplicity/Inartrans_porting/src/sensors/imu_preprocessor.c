#include "imu_preprocessor.h"

#include <string.h>

bool CookieIMU_PreprocessForNavigation(const CookieImuConvertedSample *input,
                                        CookieImuNavigationInput *output)
{
    if (input == NULL || output == NULL || !input->valid) {
        return false;
    }

    memset(output, 0, sizeof(*output));

    /*
     * Axis convention adapted from the original Inartrans_v2 implementation.
     *
     * Original mapping before EKF_Predict:
     *   acc_x = -x
     *   acc_y = -z
     *   acc_z = -y
     *
     * Same mapping is applied to gyroscope data.
     */
    output->accel_m_s2[0] = -input->accel_m_s2[0];
    output->accel_m_s2[1] = -input->accel_m_s2[2];
    output->accel_m_s2[2] = -input->accel_m_s2[1];

    output->gyro_rad_s[0] = -input->gyro_rad_s[0];
    output->gyro_rad_s[1] = -input->gyro_rad_s[2];
    output->gyro_rad_s[2] = -input->gyro_rad_s[1];

    output->dt_s = input->dt_s;
    output->valid = true;

    return true;
}
