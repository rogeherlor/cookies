



#include "gnss_converter.h"

#include <math.h>
#include <stddef.h>

static float nmea_coordinate_to_decimal(float raw_coordinate)
{
    int degrees = (int)(raw_coordinate / 100.0f);
    float minutes = raw_coordinate - (degrees * 100.0f);

    return (float)degrees + (minutes / 60.0f);
}

bool CookieGNSS_ConvertToDecimalDegrees(const CookieGnssFix *fix,
                                         float *latitude_deg,
                                         float *longitude_deg)
{
    if (fix == NULL || latitude_deg == NULL || longitude_deg == NULL) {
        return false;
    }

    if (!fix->valid) {
        return false;
    }

    if (fix->latitude_raw == 0.0f || fix->longitude_raw == 0.0f) {
        return false;
    }

    float lat = nmea_coordinate_to_decimal(fix->latitude_raw);
    float lon = nmea_coordinate_to_decimal(fix->longitude_raw);

    if (fix->latitude_dir == 'S') {
        lat = -lat;
    } else if (fix->latitude_dir != 'N') {
        return false;
    }

    if (fix->longitude_dir == 'W') {
        lon = -lon;
    } else if (fix->longitude_dir != 'E') {
        return false;
    }

    *latitude_deg = lat;
    *longitude_deg = lon;

    return true;
}

