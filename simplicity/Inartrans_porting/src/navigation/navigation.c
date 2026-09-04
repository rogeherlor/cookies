#include "navigation.h"

#include <math.h>
#include <string.h>

#include "../gnss/gnss_converter.h"

/*
 * External EKF dependency.
 *
 * The EKF implementation is maintained outside this module.
 * This wrapper only adapts project-level GNSS data to the EKF public interface.
 *
 * When this code is integrated into Simplicity Studio, the project must include:
 * - cookie_ekf.h in the include path
 * - cookie_ekf.c in the build sources
 */
// #include "cookie_ekf.h"
#include "mock_ekf.h"

#define COOKIE_NAVIGATION_VALID_GNSS_THRESHOLD 10U

static EKF_Context_t navigation_ekf;
static bool navigation_initialized = false;
static unsigned int consecutive_valid_gnss = 0U;

void CookieNavigation_Init(void)
{
    memset(&navigation_ekf, 0, sizeof(navigation_ekf));
    navigation_initialized = false;
    consecutive_valid_gnss = 0U;
}

bool CookieNavigation_IsInitialized(void)
{
    return navigation_initialized;
}

bool CookieNavigation_UpdateWithGnss(const CookieGnssFix *fix)
{
    if (fix == NULL || !fix->valid) {
        consecutive_valid_gnss = 0U;
        return false;
    }

    float latitude_deg = 0.0f;
    float longitude_deg = 0.0f;

    if (!CookieGNSS_ConvertToDecimalDegrees(fix, &latitude_deg, &longitude_deg)) {
        consecutive_valid_gnss = 0U;
        return false;
    }

    if (consecutive_valid_gnss < COOKIE_NAVIGATION_VALID_GNSS_THRESHOLD) {
        consecutive_valid_gnss++;
    }

    /*
     * The old implementation waited for several consecutive valid GNSS fixes
     * before initialising the EKF. This avoids using an unstable first fix as
     * the navigation origin.
     */
    if (!navigation_initialized) {
        if (consecutive_valid_gnss >= COOKIE_NAVIGATION_VALID_GNSS_THRESHOLD) {
            EKF_Init(&navigation_ekf, latitude_deg, longitude_deg, fix->altitude_m);
            navigation_initialized = true;
        }

        return navigation_initialized;
    }

    EKF_Update(&navigation_ekf, latitude_deg, longitude_deg, fix->altitude_m);

    return true;
}

bool CookieNavigation_PredictWithImu(const CookieImuNavigationInput *imu)
{
    if (imu == NULL || !imu->valid) {
        return false;
    }

    if (!navigation_initialized) {
        return false;
    }

    if (imu->dt_s <= 0.0f) {
        return false;
    }

    EKF_Predict(&navigation_ekf,
                (float *)imu->accel_m_s2,
                (float *)imu->gyro_rad_s,
                imu->dt_s);

    return true;
}

bool CookieNavigation_GetState(CookieNavigationState *state)
{
    if (state == NULL || !navigation_initialized) {
        return false;
    }

    float lla[3] = {0.0f, 0.0f, 0.0f};

    ENU_to_LLA(navigation_ekf.pos_enu,
               navigation_ekf.lla0[0],
               navigation_ekf.lla0[1],
               navigation_ekf.lla0[2],
               lla);

    state->valid = true;
    state->latitude_deg = lla[0];
    state->longitude_deg = lla[1];
    state->altitude_m = lla[2];

    state->velocity_m_s = sqrtf(
        navigation_ekf.vel_enu[0] * navigation_ekf.vel_enu[0] +
        navigation_ekf.vel_enu[1] * navigation_ekf.vel_enu[1] +
        navigation_ekf.vel_enu[2] * navigation_ekf.vel_enu[2]
    );

    return true;
}
