#ifndef COOKIE_APP_H
#define COOKIE_APP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../gnss/gnss.h"
#include "../sensors/imu_sample.h"
#include "../navigation/navigation.h"
#include "../packets/packet_builder.h"
#include "../network/network_frame.h"

typedef struct {
    uint8_t gnss_mode;
    CookieNetworkFrameHeader network_header;
} CookieAppConfig;

typedef struct {
    CookieAppConfig config;

    bool has_gnss;
    bool has_imu;
    bool has_navigation;

    CookieGnssFix last_gnss_fix;
    CookieImuSample last_imu_sample;
    CookieNavigationState navigation_state;
} CookieAppContext;

typedef struct {
    bool converted;
    bool first_sample;
    bool preprocessed;
    bool navigation_initialized;
    bool navigation_predicted;

    float dt_s;

    float accel_m_s2[3];
    float gyro_rad_s[3];

    float navigation_accel_m_s2[3];
    float navigation_gyro_rad_s[3];
} CookieAppImuProcessDebug;


void CookieApp_Init(CookieAppContext *app,
                    const CookieAppConfig *config);

bool CookieApp_ProcessGnssEpoch(CookieAppContext *app,
                                const char *nmea_buffer,
                                uint32_t nmea_length);

bool CookieApp_ProcessImuSample(CookieAppContext *app,
                                const int32_t accel_mg[3],
                                const int32_t gyro_dps[3],
                                uint32_t timestamp_ms);

bool CookieApp_ProcessImuSampleWithDebug(CookieAppContext *app,
                                         const int32_t accel_mg[3],
                                         const int32_t gyro_dps[3],
                                         uint32_t timestamp_ms,
                                         CookieAppImuProcessDebug *debug);

bool CookieApp_BuildDataMessage(CookieAppContext *app,
                                uint8_t *message,
                                size_t message_size);

#endif /* COOKIE_APP_H */
