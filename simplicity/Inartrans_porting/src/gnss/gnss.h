#ifndef COOKIE_GNSS_H
#define COOKIE_GNSS_H

#include <stdbool.h>
#include <stdint.h>

#define COOKIE_GNSS_TIME_LEN 11
#define COOKIE_GNSS_DATE_LEN 7

typedef struct {
    bool valid;

    char time_utc[COOKIE_GNSS_TIME_LEN];  // HHMMSS.mmm
    char date_ddmmyy[COOKIE_GNSS_DATE_LEN];

    float latitude_raw;   // NMEA format: ddmm.mmmm
    char latitude_dir;    // N or S

    float longitude_raw;  // NMEA format: dddmm.mmmm
    char longitude_dir;   // E or W

    float altitude_m;

    uint16_t speed_cm_s;
    uint16_t cog_cdeg;
    uint16_t pdop_x100;
} CookieGnssFix;

bool CookieGNSS_ParseEpoch(const char *buffer, uint32_t length, CookieGnssFix *fix);

#endif