#ifndef COOKIE_GNSS_CONVERTER_H
#define COOKIE_GNSS_CONVERTER_H

#include <stdbool.h>

#include "gnss.h"

bool CookieGNSS_ConvertToDecimalDegrees(const CookieGnssFix *fix,
                                         float *latitude_deg,
                                         float *longitude_deg);

#endif
