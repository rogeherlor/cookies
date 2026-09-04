#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "../src/packets/packet_builder.h"

// gcc \
//   simplicity/Inartrans_porting/tests/test_packet_builder.c \
//   simplicity/Inartrans_porting/src/packets/packet_builder.c \
//   -I simplicity/Inartrans_porting/src/packets \
//   -o simplicity/Inartrans_porting/tests/test_packet_builder

// ./simplicity/Inartrans_porting/tests/test_packet_builder

static void read_bytes(const uint8_t *buffer, size_t offset,
                       void *dst, size_t size)
{
    memcpy(dst, buffer + offset, size);
}

int main(void)
{
    uint8_t packet[COOKIE_PACKET_DATA_SIZE];

    CookiePacketData data = {0};

    data.legacy_environment.relative_humidity = 0;
    data.legacy_environment.temperature = 0;

    data.imu.accel_mg[0] = 100;
    data.imu.accel_mg[1] = -20;
    data.imu.accel_mg[2] = 980;

    data.link.original_link_rssi = 0;

    data.gnss.available = true;
    data.gnss.valid = true;
    data.gnss.latitude_raw = 4807.038f;
    data.gnss.latitude_direction = 'N';
    data.gnss.longitude_raw = 1131.000f;
    data.gnss.longitude_direction = 'E';
    data.gnss.altitude_m = 545.4f;
    data.gnss.speed_cm_s = 1152;
    memcpy(data.gnss.time_utc, "123519.000", 10);
    memcpy(data.gnss.date, "940323", 6);
    data.gnss.pdop_centi = 180;
    data.gnss.mode = 7;

    data.navigation.latitude_deg = 48.117302f;
    data.navigation.longitude_deg = 11.516666f;
    data.navigation.altitude_m = 545.4f;
    data.navigation.speed_m_s = 11.52f;

    assert(CookiePacket_BuildDataPacket(&data, packet, sizeof(packet)));

    assert(packet[21] == 'A');
    assert(packet[30] == 'N');
    assert(packet[39] == 'E');
    assert(packet[72] == 7);
    assert(packet[73] == ',');
    assert(packet[74] == ',');

    float nav_lat = 0.0f;
    float nav_lon = 0.0f;
    uint16_t speed = 0;
    uint16_t pdop = 0;

    read_bytes(packet, 26, &nav_lat, sizeof(nav_lat));
    read_bytes(packet, 35, &nav_lon, sizeof(nav_lon));
    read_bytes(packet, 48, &speed, sizeof(speed));
    read_bytes(packet, 70, &pdop, sizeof(pdop));

    assert(nav_lat > 48.117f && nav_lat < 48.118f);
    assert(nav_lon > 11.516f && nav_lon < 11.517f);
    assert(speed == 1152);
    assert(pdop == 180);

    printf("Packet builder test passed\n");
    printf("Packet size: %u bytes\n", (unsigned)COOKIE_PACKET_DATA_SIZE);
    printf("Validity: %c\n", packet[21]);
    printf("GNSS mode: %u\n", packet[72]);

    return 0;
}