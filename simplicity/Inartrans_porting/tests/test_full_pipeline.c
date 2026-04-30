// gcc \
//   simplicity/Inartrans_porting/tests/test_full_pipeline.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   simplicity/Inartrans_porting/src/gnss/gnss_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/navigation/navigation.c \
//   simplicity/Inartrans_porting/src/navigation/mock_ekf.c \
//   simplicity/Inartrans_porting/src/packets/packet_builder.c \
//   simplicity/Inartrans_porting/src/network/network_frame.c \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -I simplicity/Inartrans_porting/src/navigation \
//   -I simplicity/Inartrans_porting/src/packets \
//   -I simplicity/Inartrans_porting/src/network \
//   -lm \
//   -o simplicity/Inartrans_porting/tests/test_full_pipeline

// ./simplicity/Inartrans_porting/tests/test_full_pipeline

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/gnss/gnss.h"
#include "../src/sensors/imu_sample.h"
#include "../src/sensors/imu_converter.h"
#include "../src/sensors/imu_preprocessor.h"
#include "../src/navigation/navigation.h"
#include "../src/packets/packet_builder.h"
#include "../src/network/network_frame.h"

static void fill_packet_data_from_modules(CookiePacketData *packet_data,
                                          const CookieGnssFix *fix,
                                          const CookieImuSample *imu_sample,
                                          const CookieNavigationState *navigation_state)
{
    memset(packet_data, 0, sizeof(*packet_data));

    packet_data->legacy_environment.relative_humidity = 0;
    packet_data->legacy_environment.temperature = 0;

    /*
     * Legacy packet stores IMU acceleration in raw milli-g values.
     */
    packet_data->imu.accel_mg[0] = imu_sample->accel_mg[0];
    packet_data->imu.accel_mg[1] = imu_sample->accel_mg[1];
    packet_data->imu.accel_mg[2] = imu_sample->accel_mg[2];

    /*
     * In the real multi-hop network this may be overwritten by relay nodes.
     * In this PC test it is kept as the source-node initial value.
     */
    packet_data->link.original_link_rssi = 0;

    packet_data->gnss.available = true;
    packet_data->gnss.valid = fix->valid;
    packet_data->gnss.latitude_raw = fix->latitude_raw;
    packet_data->gnss.latitude_direction = fix->latitude_dir;
    packet_data->gnss.longitude_raw = fix->longitude_raw;
    packet_data->gnss.longitude_direction = fix->longitude_dir;
    packet_data->gnss.altitude_m = fix->altitude_m;
    packet_data->gnss.speed_cm_s = fix->speed_cm_s;
    packet_data->gnss.pdop_centi = fix->pdop_x100;

    /*
     * CookieGnssFix stores date as DDMMYY.
     * The legacy packet stores date as YYMMDD.
     */
    for (int i = 0; i < 10; i++) {
        packet_data->gnss.time_utc[i] = fix->time_utc[i];
    }

    packet_data->gnss.date[0] = fix->date_ddmmyy[4];
    packet_data->gnss.date[1] = fix->date_ddmmyy[5];
    packet_data->gnss.date[2] = fix->date_ddmmyy[2];
    packet_data->gnss.date[3] = fix->date_ddmmyy[3];
    packet_data->gnss.date[4] = fix->date_ddmmyy[0];
    packet_data->gnss.date[5] = fix->date_ddmmyy[1];

    /*
     * Placeholder for now. In the real application this will come from the
     * GNSS configuration/state logic.
     */
    packet_data->gnss.mode = 7;

    packet_data->navigation.latitude_deg = navigation_state->latitude_deg;
    packet_data->navigation.longitude_deg = navigation_state->longitude_deg;
    packet_data->navigation.altitude_m = navigation_state->altitude_m;
    packet_data->navigation.speed_m_s = navigation_state->velocity_m_s;
}

static void print_full_pipeline_summary(const CookiePacketData *packet_data,
                                        const uint8_t packet[COOKIE_PACKET_DATA_SIZE],
                                        const uint8_t message[COOKIE_NETWORK_DATA_FRAME_SIZE])
{
    printf("Full pipeline test passed\n");

    printf("\nData packet\n");
    printf("Packet size: %u bytes\n", (unsigned)COOKIE_PACKET_DATA_SIZE);
    printf("Packet validity byte: %c\n", packet[21]);
    printf("Packet GNSS mode byte: %u\n", packet[72]);
    printf("Packet end markers: %c%c\n", packet[73], packet[74]);

    printf("\nNetwork frame\n");
    printf("Header size: %u bytes\n", (unsigned)COOKIE_NETWORK_HEADER_SIZE);
    printf("Frame size: %u bytes\n", (unsigned)COOKIE_NETWORK_DATA_FRAME_SIZE);
    printf("Frame packet type: %u\n", message[0]);
    printf("Payload starts at message[15]: %u\n", message[15]);
    printf("Payload ends at message[89]: %u\n", message[89]);

    printf("\nDecoded logical data\n");
    printf("GNSS raw lat: %.3f %c\n",
           packet_data->gnss.latitude_raw,
           packet_data->gnss.latitude_direction);
    printf("GNSS raw lon: %.3f %c\n",
           packet_data->gnss.longitude_raw,
           packet_data->gnss.longitude_direction);
    printf("GNSS altitude: %.2f m\n", packet_data->gnss.altitude_m);
    printf("GNSS speed: %u cm/s\n", packet_data->gnss.speed_cm_s);
    printf("PDOP: %u\n", packet_data->gnss.pdop_centi);
    printf("Navigation lat: %.6f deg\n", packet_data->navigation.latitude_deg);
    printf("Navigation lon: %.6f deg\n", packet_data->navigation.longitude_deg);
    printf("Navigation alt: %.2f m\n", packet_data->navigation.altitude_m);
    printf("Navigation speed: %.3f m/s\n", packet_data->navigation.speed_m_s);
}

int main(void)
{
    CookieNavigation_Init();
    CookieIMU_ConverterReset();

    /*
     * 1. GNSS parsing
     */
    CookieGnssFix fix = {0};

    const char *nmea_epoch =
        "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A*62\r\n"
        "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59\r\n"
        "$GNGSA,A,3,04,05,09,12,24,25,29,31,,,,,1.8,1.0,1.5*23\r\n";

    bool parsed = CookieGNSS_ParseEpoch(nmea_epoch, strlen(nmea_epoch), &fix);
    assert(parsed);
    assert(fix.valid);

    /*
     * 2. Navigation initialisation/update using several valid GNSS fixes.
     */
    for (int i = 0; i < 10; i++) {
        bool updated = CookieNavigation_UpdateWithGnss(&fix);
        printf("GNSS update %d -> %s\n", i + 1, updated ? "true" : "false");
    }

    assert(CookieNavigation_IsInitialized());

    /*
     * 3. IMU conversion and preprocessing.
     * The first sample only initializes the converter timestamp.
     * The second sample provides a positive dt for prediction.
     */
    CookieImuSample imu_sample = {0};

    int32_t accel_mg[3] = {100, -20, 980};
    int32_t gyro_dps[3] = {1, 2, -3};

    CookieIMU_SetSample(&imu_sample, accel_mg, gyro_dps, 1000);
    assert(imu_sample.valid);

    CookieImuConvertedSample imu_converted = {0};
    CookieImuNavigationInput imu_navigation = {0};

    bool converted = CookieIMU_ConvertSample(&imu_sample, &imu_converted);
    assert(converted);

    CookieIMU_SetSample(&imu_sample, accel_mg, gyro_dps, 1010);
    assert(imu_sample.valid);

    converted = CookieIMU_ConvertSample(&imu_sample, &imu_converted);
    assert(converted);
    assert(imu_converted.dt_s > 0.0f);

    bool preprocessed = CookieIMU_PreprocessForNavigation(&imu_converted, &imu_navigation);
    assert(preprocessed);
    assert(imu_navigation.dt_s > 0.0f);

    /*
     * 4. Navigation prediction.
     */
    bool predicted = CookieNavigation_PredictWithImu(&imu_navigation);
    assert(predicted);

    CookieNavigationState navigation_state = {0};
    bool has_state = CookieNavigation_GetState(&navigation_state);
    assert(has_state);
    assert(navigation_state.valid);

    /*
     * 5. Logical packet data.
     */
    CookiePacketData packet_data = {0};

    fill_packet_data_from_modules(&packet_data,
                                  &fix,
                                  &imu_sample,
                                  &navigation_state);

    /*
     * 6. Legacy 75-byte data packet.
     */
    uint8_t packet[COOKIE_PACKET_DATA_SIZE] = {0};

    bool packet_built = CookiePacket_BuildDataPacket(&packet_data,
                                                     packet,
                                                     sizeof(packet));

    assert(packet_built);

    assert(packet[21] == 'A');
    assert(packet[30] == 'N');
    assert(packet[39] == 'E');
    assert(packet[72] == 7);
    assert(packet[73] == ',');
    assert(packet[74] == ',');

    /*
     * 7. Legacy 90-byte network frame.
     */
    CookieNetworkFrameHeader header = {0};

    header.packet_type = COOKIE_PACKET_TYPE_DATA;
    header.sender_rank = 1;
    header.destination = 0x0000;
    header.pan_id = 0x1234;
    header.source = 0xABCD;
    header.packet_number = 42;
    header.source_rank = 1;
    header.sequence = 7;

    uint8_t message[COOKIE_NETWORK_DATA_FRAME_SIZE] = {0};

    bool frame_built = CookieNetwork_BuildDataFrame(&header,
                                                    packet,
                                                    sizeof(packet),
                                                    message,
                                                    sizeof(message));

    assert(frame_built);

    assert(message[0] == COOKIE_PACKET_TYPE_DATA);
    assert(message[COOKIE_NETWORK_HEADER_SIZE] == packet[0]);
    assert(message[COOKIE_NETWORK_DATA_FRAME_SIZE - 1] == packet[COOKIE_PACKET_DATA_SIZE - 1]);

    print_full_pipeline_summary(&packet_data, packet, message);

    return 0;
}