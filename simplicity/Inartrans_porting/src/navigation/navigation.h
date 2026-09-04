#ifndef COOKIE_NAVIGATION_H
#define COOKIE_NAVIGATION_H

#include <stdbool.h>

#include "../gnss/gnss.h"
#include "../sensors/imu_preprocessor.h"

typedef struct {
    bool valid;

    float latitude_deg;
    float longitude_deg;
    float altitude_m;

    float velocity_m_s;
} CookieNavigationState;

void CookieNavigation_Init(void);

bool CookieNavigation_IsInitialized(void);

bool CookieNavigation_UpdateWithGnss(const CookieGnssFix *fix);

bool CookieNavigation_PredictWithImu(const CookieImuNavigationInput *imu);

bool CookieNavigation_GetState(CookieNavigationState *state);

#endif
