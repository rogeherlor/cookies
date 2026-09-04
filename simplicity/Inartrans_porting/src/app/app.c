#include "app.h"

#include <string.h>

#include "../sensors/imu_converter.h"
#include "../sensors/imu_preprocessor.h"

static void fill_packet_data(CookiePacketData *packet_data,
                             const CookieAppContext *app)
{
    memset(packet_data, 0, sizeof(*packet_data));

    packet_data->legacy_environment.relative_humidity = 0;
    packet_data->legacy_environment.temperature = 0;

    /*
     * Legacy packet stores raw IMU acceleration in milli-g.
     */
    packet_data->imu.accel_mg[0] = app->last_imu_sample.accel_mg[0];
    packet_data->imu.accel_mg[1] = app->last_imu_sample.accel_mg[1];
    packet_data->imu.accel_mg[2] = app->last_imu_sample.accel_mg[2];

    /*
     * In the real multi-hop network this field may be overwritten by relay
     * nodes. At the source node it starts as zero.
     */
    packet_data->link.original_link_rssi = 0;

    packet_data->gnss.available = app->has_gnss;
    packet_data->gnss.valid = app->last_gnss_fix.valid;

    packet_data->gnss.latitude_raw = app->last_gnss_fix.latitude_raw;
    packet_data->gnss.latitude_direction = app->last_gnss_fix.latitude_dir;

    packet_data->gnss.longitude_raw = app->last_gnss_fix.longitude_raw;
    packet_data->gnss.longitude_direction = app->last_gnss_fix.longitude_dir;

    packet_data->gnss.altitude_m = app->last_gnss_fix.altitude_m;
    packet_data->gnss.speed_cm_s = app->last_gnss_fix.speed_cm_s;
    packet_data->gnss.pdop_centi = app->last_gnss_fix.pdop_x100;
    packet_data->gnss.mode = app->config.gnss_mode;

    /*
     * CookieGnssFix stores time as HHMMSS.mmm plus string terminator.
     * The legacy packet stores only the first 10 bytes.
     */
    for (int i = 0; i < 10; i++) {
        packet_data->gnss.time_utc[i] = app->last_gnss_fix.time_utc[i];
    }

    /*
     * CookieGnssFix stores date as DDMMYY.
     * The legacy packet stores date as YYMMDD.
     */
    packet_data->gnss.date[0] = app->last_gnss_fix.date_ddmmyy[4];
    packet_data->gnss.date[1] = app->last_gnss_fix.date_ddmmyy[5];
    packet_data->gnss.date[2] = app->last_gnss_fix.date_ddmmyy[2];
    packet_data->gnss.date[3] = app->last_gnss_fix.date_ddmmyy[3];
    packet_data->gnss.date[4] = app->last_gnss_fix.date_ddmmyy[0];
    packet_data->gnss.date[5] = app->last_gnss_fix.date_ddmmyy[1];

    packet_data->navigation.latitude_deg = app->navigation_state.latitude_deg;
    packet_data->navigation.longitude_deg = app->navigation_state.longitude_deg;
    packet_data->navigation.altitude_m = app->navigation_state.altitude_m;
    packet_data->navigation.speed_m_s = app->navigation_state.velocity_m_s;
}

void CookieApp_Init(CookieAppContext *app,
                    const CookieAppConfig *config)
{
    if (app == NULL) {
        return;
    }

    memset(app, 0, sizeof(*app));

    if (config != NULL) {
        app->config = *config;
    }

    CookieNavigation_Init();
    CookieIMU_ConverterReset();
}

bool CookieApp_ProcessGnssEpoch(CookieAppContext *app,
                                const char *nmea_buffer,
                                uint32_t nmea_length)
{
    if (app == NULL || nmea_buffer == NULL) {
        return false;
    }

    CookieGnssFix fix = {0};

    bool parsed = CookieGNSS_ParseEpoch(nmea_buffer, nmea_length, &fix);
    if (!parsed) {
        return false;
    }

    app->last_gnss_fix = fix;
    app->has_gnss = true;

    (void)CookieNavigation_UpdateWithGnss(&fix);

    if (CookieNavigation_IsInitialized()) {
        CookieNavigationState state = {0};

        if (CookieNavigation_GetState(&state)) {
            app->navigation_state = state;
            app->has_navigation = state.valid;
        }
    }

    return true;
}

bool CookieApp_ProcessImuSample(CookieAppContext *app,
                                const int32_t accel_mg[3],
                                const int32_t gyro_dps[3],
                                uint32_t timestamp_ms)
{
    return CookieApp_ProcessImuSampleWithDebug(app,
                                               accel_mg,
                                               gyro_dps,
                                               timestamp_ms,
                                               NULL);
}

bool CookieApp_ProcessImuSampleWithDebug(CookieAppContext *app,
                                         const int32_t accel_mg[3],
                                         const int32_t gyro_dps[3],
                                         uint32_t timestamp_ms,
                                         CookieAppImuProcessDebug *debug)
{
    if (debug != NULL) {
        memset(debug, 0, sizeof(*debug));
    }

    if (app == NULL || accel_mg == NULL || gyro_dps == NULL) {
        return false;
    }

    CookieImuSample sample = {0};
    CookieIMU_SetSample(&sample, accel_mg, gyro_dps, timestamp_ms);

    app->last_imu_sample = sample;
    app->has_imu = sample.valid;

    CookieImuConvertedSample converted = {0};

    bool converted_ok = CookieIMU_ConvertSample(&sample, &converted);
    if (!converted_ok) {
        return false;
    }

    if (debug != NULL) {
        debug->converted = true;
        debug->dt_s = converted.dt_s;

        memcpy(debug->accel_m_s2,
               converted.accel_m_s2,
               sizeof(debug->accel_m_s2));

        memcpy(debug->gyro_rad_s,
               converted.gyro_rad_s,
               sizeof(debug->gyro_rad_s));
    }

    /*
     * The first IMU sample after reset only initializes the timestamp.
     * It has dt_s = 0 and should not trigger EKF prediction.
     */
    if (converted.dt_s <= 0.0f) {
        if (debug != NULL) {
            debug->first_sample = true;
        }

        return true;
    }

    CookieImuNavigationInput navigation_input = {0};

    bool preprocessed = CookieIMU_PreprocessForNavigation(&converted,
                                                          &navigation_input);
    if (!preprocessed) {
        return false;
    }

    if (debug != NULL) {
        debug->preprocessed = true;

        memcpy(debug->navigation_accel_m_s2,
               navigation_input.accel_m_s2,
               sizeof(debug->navigation_accel_m_s2));

        memcpy(debug->navigation_gyro_rad_s,
               navigation_input.gyro_rad_s,
               sizeof(debug->navigation_gyro_rad_s));
    }

    bool navigation_ready = CookieNavigation_IsInitialized();

    if (debug != NULL) {
        debug->navigation_initialized = navigation_ready;
    }

    if (navigation_ready) {
        bool predicted = CookieNavigation_PredictWithImu(&navigation_input);

        if (debug != NULL) {
            debug->navigation_predicted = predicted;
        }

        if (predicted) {
            CookieNavigationState state = {0};

            if (CookieNavigation_GetState(&state)) {
                app->navigation_state = state;
                app->has_navigation = state.valid;
            }
        }
    }

    return true;
}

bool CookieApp_BuildDataMessage(CookieAppContext *app,
                                uint8_t *message,
                                size_t message_size)
{
    if (app == NULL || message == NULL) {
        return false;
    }

    if (!app->has_gnss || !app->has_imu || !app->has_navigation) {
        return false;
    }

    CookiePacketData packet_data = {0};
    uint8_t packet[COOKIE_PACKET_DATA_SIZE] = {0};

    fill_packet_data(&packet_data, app);

    bool packet_built = CookiePacket_BuildDataPacket(&packet_data,
                                                     packet,
                                                     sizeof(packet));

    if (!packet_built) {
        return false;
    }

    return CookieNetwork_BuildDataFrame(&app->config.network_header,
                                        packet,
                                        sizeof(packet),
                                        message,
                                        message_size);
}
