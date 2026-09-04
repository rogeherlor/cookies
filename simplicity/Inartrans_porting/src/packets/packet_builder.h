#ifndef COOKIE_PACKET_BUILDER_H
#define COOKIE_PACKET_BUILDER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define COOKIE_PACKET_DATA_SIZE 75U

typedef struct {
    uint32_t relative_humidity;
    int32_t temperature;
} CookiePacketLegacyEnvironment;

typedef struct {
    int32_t accel_mg[3];
} CookiePacketImuData;

typedef struct {
    int8_t original_link_rssi;
} CookiePacketLinkData;

typedef struct {
    bool available;
    bool valid;

    float latitude_raw;
    char latitude_direction;

    float longitude_raw;
    char longitude_direction;

    float altitude_m;
    uint16_t speed_cm_s;

    char time_utc[10];   /* "HHMMSS.mmm" */
    char date[6];        /* "YYMMDD" */

    uint16_t pdop_centi;
    uint8_t mode;
} CookiePacketGnssData;

typedef struct {
    float latitude_deg;
    float longitude_deg;
    float altitude_m;
    float speed_m_s;
} CookiePacketNavigationData;

typedef struct {
    CookiePacketLegacyEnvironment legacy_environment;
    CookiePacketImuData imu;
    CookiePacketLinkData link;
    CookiePacketGnssData gnss;
    CookiePacketNavigationData navigation;
} CookiePacketData;

bool CookiePacket_BuildDataPacket(const CookiePacketData *data,
                                  uint8_t *buffer,
                                  size_t buffer_size);

#endif /* COOKIE_PACKET_BUILDER_H */