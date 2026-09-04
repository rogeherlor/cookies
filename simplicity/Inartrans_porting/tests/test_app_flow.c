// gcc \
//   simplicity/Inartrans_porting/tests/test_app_flow.c \
//   simplicity/Inartrans_porting/src/app/app.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   simplicity/Inartrans_porting/src/gnss/gnss_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_sample.c \
//   simplicity/Inartrans_porting/src/sensors/imu_converter.c \
//   simplicity/Inartrans_porting/src/sensors/imu_preprocessor.c \
//   simplicity/Inartrans_porting/src/navigation/navigation.c \
//   simplicity/Inartrans_porting/src/navigation/mock_ekf.c \
//   simplicity/Inartrans_porting/src/packets/packet_builder.c \
//   simplicity/Inartrans_porting/src/network/network_frame.c \
//   -I simplicity/Inartrans_porting/src/app \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -I simplicity/Inartrans_porting/src/sensors \
//   -I simplicity/Inartrans_porting/src/navigation \
//   -I simplicity/Inartrans_porting/src/packets \
//   -I simplicity/Inartrans_porting/src/network \
//   -lm \
//   -o simplicity/Inartrans_porting/tests/test_app_flow

// ./simplicity/Inartrans_porting/tests/test_app_flow

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/app/app.h"

int main(void)
{
    CookieAppConfig config = {0};

    config.gnss_mode = 7;

    config.network_header.packet_type = COOKIE_PACKET_TYPE_DATA;
    config.network_header.sender_rank = 1;
    config.network_header.destination = 0x0000;
    config.network_header.pan_id = 0x1234;
    config.network_header.source = 0xABCD;
    config.network_header.packet_number = 42;
    config.network_header.source_rank = 1;
    config.network_header.sequence = 7;

    CookieAppContext app = {0};
    CookieApp_Init(&app, &config);

    const char *nmea_epoch =
        "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A*62\r\n"
        "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59\r\n"
        "$GNGSA,A,3,04,05,09,12,24,25,29,31,,,,,1.8,1.0,1.5*23\r\n";

    /*
     * Navigation waits for 10 valid GNSS fixes before becoming initialized.
     */
    for (int i = 0; i < 10; i++) {
        bool gnss_ok = CookieApp_ProcessGnssEpoch(&app,
                                                  nmea_epoch,
                                                  strlen(nmea_epoch));
        assert(gnss_ok);
        printf("App GNSS epoch %d processed\n", i + 1);
    }

    assert(app.has_gnss);
    assert(app.has_navigation);

    int32_t accel_mg[3] = {100, -20, 980};
    int32_t gyro_dps[3] = {1, 2, -3};

    /*
     * First IMU sample initializes converter timestamp.
     */
    bool imu_ok = CookieApp_ProcessImuSample(&app,
                                             accel_mg,
                                             gyro_dps,
                                             1000);
    assert(imu_ok);

    /*
     * Second IMU sample produces positive dt and allows prediction.
     */
    imu_ok = CookieApp_ProcessImuSample(&app,
                                        accel_mg,
                                        gyro_dps,
                                        1010);
    assert(imu_ok);

    assert(app.has_imu);
    assert(app.has_navigation);

    uint8_t message[COOKIE_NETWORK_DATA_FRAME_SIZE] = {0};

    bool message_built = CookieApp_BuildDataMessage(&app,
                                                    message,
                                                    sizeof(message));

    assert(message_built);

    assert(message[0] == COOKIE_PACKET_TYPE_DATA);
    assert(message[15 + 21] == 'A');
    assert(message[15 + 30] == 'N');
    assert(message[15 + 39] == 'E');
    assert(message[15 + 72] == 7);
    assert(message[15 + 73] == ',');
    assert(message[15 + 74] == ',');

    printf("App flow test passed\n");
    printf("Message size: %u bytes\n", (unsigned)COOKIE_NETWORK_DATA_FRAME_SIZE);
    printf("Packet type: %u\n", message[0]);
    printf("Payload validity byte: %c\n", message[15 + 21]);
    printf("Payload GNSS mode byte: %u\n", message[15 + 72]);
    printf("Payload end markers: %c%c\n", message[15 + 73], message[15 + 74]);

    return 0;
}