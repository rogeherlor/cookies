#include "packet_builder.h"

#include <string.h>

/*
 * Legacy Inartrans_v2 75-byte data packet layout.
 *
 * This module preserves the old binary layout, but keeps the offsets isolated
 * from the rest of the application.
 */

#define OFF_RH_DATA          0U
#define OFF_TEMP_DATA        4U
#define OFF_ACCEL_X          8U
#define OFF_ACCEL_Y          12U
#define OFF_ACCEL_Z          16U
#define OFF_ORIGINAL_RSSI    20U
#define OFF_GNSS_VALIDITY    21U
#define OFF_GNSS_LAT_RAW     22U
#define OFF_NAV_LAT          26U
#define OFF_GNSS_NS          30U
#define OFF_GNSS_LON_RAW     31U
#define OFF_NAV_LON          35U
#define OFF_GNSS_EW          39U
#define OFF_GNSS_ALT         40U
#define OFF_NAV_ALT          44U
#define OFF_GNSS_SPEED       48U
#define OFF_NAV_SPEED        50U
#define OFF_GNSS_TIME        54U
#define OFF_GNSS_DATE        64U
#define OFF_GNSS_PDOP        70U
#define OFF_GNSS_MODE        72U
#define OFF_END_MARKER_0     73U
#define OFF_END_MARKER_1     74U

static void write_bytes(uint8_t *buffer, size_t offset,
                        const void *src, size_t size)
{
    memcpy(buffer + offset, src, size);
}

bool CookiePacket_BuildDataPacket(const CookiePacketData *data,
                                  uint8_t *buffer,
                                  size_t buffer_size)
{
    if (data == NULL || buffer == NULL) {
        return false;
    }

    if (buffer_size < COOKIE_PACKET_DATA_SIZE) {
        return false;
    }

    memset(buffer, 0, COOKIE_PACKET_DATA_SIZE);

    write_bytes(buffer, OFF_RH_DATA,
                &data->legacy_environment.relative_humidity,
                sizeof(data->legacy_environment.relative_humidity));

    write_bytes(buffer, OFF_TEMP_DATA,
                &data->legacy_environment.temperature,
                sizeof(data->legacy_environment.temperature));

    write_bytes(buffer, OFF_ACCEL_X,
                &data->imu.accel_mg[0],
                sizeof(data->imu.accel_mg[0]));

    write_bytes(buffer, OFF_ACCEL_Y,
                &data->imu.accel_mg[1],
                sizeof(data->imu.accel_mg[1]));

    write_bytes(buffer, OFF_ACCEL_Z,
                &data->imu.accel_mg[2],
                sizeof(data->imu.accel_mg[2]));

    write_bytes(buffer, OFF_ORIGINAL_RSSI,
                &data->link.original_link_rssi,
                sizeof(data->link.original_link_rssi));

    write_bytes(buffer, OFF_NAV_LAT,
                &data->navigation.latitude_deg,
                sizeof(data->navigation.latitude_deg));

    write_bytes(buffer, OFF_NAV_LON,
                &data->navigation.longitude_deg,
                sizeof(data->navigation.longitude_deg));

    write_bytes(buffer, OFF_NAV_ALT,
                &data->navigation.altitude_m,
                sizeof(data->navigation.altitude_m));

    write_bytes(buffer, OFF_NAV_SPEED,
                &data->navigation.speed_m_s,
                sizeof(data->navigation.speed_m_s));

    write_bytes(buffer, OFF_GNSS_MODE,
                &data->gnss.mode,
                sizeof(data->gnss.mode));

    buffer[OFF_END_MARKER_0] = ',';
    buffer[OFF_END_MARKER_1] = ',';

    /*
     * Legacy behaviour:
     * - If GNSS is not available, validity is transmitted as 'V'.
     * - Full GNSS fields are only written when the fix is valid.
     */
    if (!data->gnss.available || !data->gnss.valid) {
        buffer[OFF_GNSS_VALIDITY] = 'V';
        return true;
    }

    buffer[OFF_GNSS_VALIDITY] = 'A';

    write_bytes(buffer, OFF_GNSS_LAT_RAW,
                &data->gnss.latitude_raw,
                sizeof(data->gnss.latitude_raw));

    write_bytes(buffer, OFF_GNSS_NS,
                &data->gnss.latitude_direction,
                sizeof(data->gnss.latitude_direction));

    write_bytes(buffer, OFF_GNSS_LON_RAW,
                &data->gnss.longitude_raw,
                sizeof(data->gnss.longitude_raw));

    write_bytes(buffer, OFF_GNSS_EW,
                &data->gnss.longitude_direction,
                sizeof(data->gnss.longitude_direction));

    write_bytes(buffer, OFF_GNSS_ALT,
                &data->gnss.altitude_m,
                sizeof(data->gnss.altitude_m));

    write_bytes(buffer, OFF_GNSS_SPEED,
                &data->gnss.speed_cm_s,
                sizeof(data->gnss.speed_cm_s));

    write_bytes(buffer, OFF_GNSS_TIME,
                data->gnss.time_utc,
                sizeof(data->gnss.time_utc));

    write_bytes(buffer, OFF_GNSS_DATE,
                data->gnss.date,
                sizeof(data->gnss.date));

    write_bytes(buffer, OFF_GNSS_PDOP,
                &data->gnss.pdop_centi,
                sizeof(data->gnss.pdop_centi));

    return true;
}