

// gcc \
//   simplicity/Inartrans_porting/tests/test_packet_integration.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   simplicity/Inartrans_porting/src/gnss/gnss_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/navigation/navigation.c \
//   simplicity/Inartrans_porting/src/navigation/mock_ekf.c \
//   simplicity/Inartrans_porting/src/packets/packet_builder.c \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -I simplicity/Inartrans_porting/src/navigation \
//   -I simplicity/Inartrans_porting/src/packets \
//   -lm \
//   -o simplicity/Inartrans_porting/tests/test_packet_integration

// ./simplicity/Inartrans_porting/tests/test_packet_integration

/*
 * NOTE:
 * The output packet replicates the legacy format, including:
 * - fixed byte offsets
 * - comma-separated fields
 * - trailing empty fields (",,")
 *
 * This is intentional for compatibility with the original system.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/gnss/gnss.h"
#include "../src/sensors/imu_sample.h"
#include "../src/sensors/imu_converter.h"
#include "../src/sensors/imu_preprocessor.h"
#include "../src/navigation/navigation.h"
#include "../src/packets/packet_builder.h"

static void print_packet_summary(const CookiePacketData *data,
                                 const uint8_t packet[COOKIE_PACKET_DATA_SIZE])
{
    printf("Packet integration test passed\n");
    printf("Packet size: %u bytes\n", (unsigned)COOKIE_PACKET_DATA_SIZE);

    printf("Packet validity byte: %c\n", packet[21]);
    printf("Packet GNSS mode byte: %u\n", packet[72]);
    printf("End markers: %c%c\n", packet[73], packet[74]);

    printf("GNSS raw lat: %.3f %c\n",
           data->gnss.latitude_raw,
           data->gnss.latitude_direction);

    printf("GNSS raw lon: %.3f %c\n",
           data->gnss.longitude_raw,
           data->gnss.longitude_direction);

    printf("GNSS altitude: %.2f m\n", data->gnss.altitude_m);
    printf("GNSS speed: %u cm/s\n", data->gnss.speed_cm_s);
    printf("PDOP: %u\n", data->gnss.pdop_centi);

    printf("Navigation lat: %.6f deg\n", data->navigation.latitude_deg);
    printf("Navigation lon: %.6f deg\n", data->navigation.longitude_deg);
    printf("Navigation alt: %.2f m\n", data->navigation.altitude_m);
    printf("Navigation speed: %.3f m/s\n", data->navigation.speed_m_s);
}

int main(void)
{
    CookieNavigation_Init();
    CookieIMU_ConverterReset();

    CookieGnssFix fix = {0};

    const char *nmea_epoch =
        "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A*62\r\n"
        "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59\r\n"
        "$GNGSA,A,3,04,05,09,12,24,25,29,31,,,,,1.8,1.0,1.5*23\r\n";

    bool parsed = CookieGNSS_ParseEpoch(nmea_epoch, strlen(nmea_epoch), &fix);
    assert(parsed);
    assert(fix.valid);

    for (int i = 0; i < 10; i++) {
        bool updated = CookieNavigation_UpdateWithGnss(&fix);
        printf("GNSS update %d -> %s\n", i + 1, updated ? "true" : "false");
    }
    
    assert(CookieNavigation_IsInitialized());

    CookieImuSample imu_sample = {0};

    int32_t accel_mg[3] = {100, -20, 980};
    int32_t gyro_dps[3] = {1, 2, -3};

    CookieIMU_SetSample(&imu_sample, accel_mg, gyro_dps, 1000);
    assert(imu_sample.valid);
    
    CookieImuConvertedSample imu_converted = {0};
    CookieImuNavigationInput imu_navigation = {0};
    
    /*
     * First IMU sample only initializes the converter timestamp.
     * Its dt_s is expected to be zero, so it is not useful for EKF prediction.
     */
    bool converted = CookieIMU_ConvertSample(&imu_sample, &imu_converted);
    assert(converted);
    
    /*
     * Second IMU sample provides a positive dt_s.
     */
    CookieIMU_SetSample(&imu_sample, accel_mg, gyro_dps, 1010);
    assert(imu_sample.valid);
    
    converted = CookieIMU_ConvertSample(&imu_sample, &imu_converted);
    assert(converted);
    assert(imu_converted.dt_s > 0.0f);
    
    bool preprocessed = CookieIMU_PreprocessForNavigation(&imu_converted, &imu_navigation);
    assert(preprocessed);
    assert(imu_navigation.dt_s > 0.0f);
    
    bool predicted = CookieNavigation_PredictWithImu(&imu_navigation);
    assert(predicted);

    CookieNavigationState navigation_state = {0};
    bool has_state = CookieNavigation_GetState(&navigation_state);
    assert(has_state);
    assert(navigation_state.valid);

    CookiePacketData packet_data = {0};

    packet_data.legacy_environment.relative_humidity = 0;
    packet_data.legacy_environment.temperature = 0;

    packet_data.imu.accel_mg[0] = imu_sample.accel_mg[0];
    packet_data.imu.accel_mg[1] = imu_sample.accel_mg[1];
    packet_data.imu.accel_mg[2] = imu_sample.accel_mg[2];

    packet_data.link.original_link_rssi = 0;

    packet_data.gnss.available = true;
    packet_data.gnss.valid = fix.valid;
    packet_data.gnss.latitude_raw = fix.latitude_raw;
    packet_data.gnss.latitude_direction = fix.latitude_dir;
    packet_data.gnss.longitude_raw = fix.longitude_raw;
    packet_data.gnss.longitude_direction = fix.longitude_dir;
    packet_data.gnss.altitude_m = fix.altitude_m;
    packet_data.gnss.speed_cm_s = fix.speed_cm_s;

    /*
     * CookieGnssFix stores date as DDMMYY.
     * The legacy packet stores date as YYMMDD.
     */
    packet_data.gnss.time_utc[0] = fix.time_utc[0];
    packet_data.gnss.time_utc[1] = fix.time_utc[1];
    packet_data.gnss.time_utc[2] = fix.time_utc[2];
    packet_data.gnss.time_utc[3] = fix.time_utc[3];
    packet_data.gnss.time_utc[4] = fix.time_utc[4];
    packet_data.gnss.time_utc[5] = fix.time_utc[5];
    packet_data.gnss.time_utc[6] = fix.time_utc[6];
    packet_data.gnss.time_utc[7] = fix.time_utc[7];
    packet_data.gnss.time_utc[8] = fix.time_utc[8];
    packet_data.gnss.time_utc[9] = fix.time_utc[9];

    packet_data.gnss.date[0] = fix.date_ddmmyy[4];
    packet_data.gnss.date[1] = fix.date_ddmmyy[5];
    packet_data.gnss.date[2] = fix.date_ddmmyy[2];
    packet_data.gnss.date[3] = fix.date_ddmmyy[3];
    packet_data.gnss.date[4] = fix.date_ddmmyy[0];
    packet_data.gnss.date[5] = fix.date_ddmmyy[1];

    packet_data.gnss.pdop_centi = fix.pdop_x100;
    packet_data.gnss.mode = 7;

    packet_data.navigation.latitude_deg = navigation_state.latitude_deg;
    packet_data.navigation.longitude_deg = navigation_state.longitude_deg;
    packet_data.navigation.altitude_m = navigation_state.altitude_m;
    packet_data.navigation.speed_m_s = navigation_state.velocity_m_s;

    uint8_t packet[COOKIE_PACKET_DATA_SIZE] = {0};

    bool built = CookiePacket_BuildDataPacket(&packet_data, packet, sizeof(packet));
    assert(built);

    assert(packet[21] == 'A');
    assert(packet[30] == 'N');
    assert(packet[39] == 'E');
    assert(packet[72] == 7);
    assert(packet[73] == ',');
    assert(packet[74] == ',');

    print_packet_summary(&packet_data, packet);

    return 0;
}