#include "gnss.h"

#include <stdlib.h>
#include <string.h>

static int hex_to_int(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }

    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }

    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }

    return -1;
}

static bool nmea_checksum_ok(const char *sentence, int length)
{
    int calculated = 0;
    int i = 0;

    while (i < length && sentence[i] != '*') {
        calculated ^= (unsigned char)sentence[i];
        i++;
    }

    if (i + 2 >= length) {
        return false;
    }

    int high = hex_to_int(sentence[i + 1]);
    int low = hex_to_int(sentence[i + 2]);

    if (high < 0 || low < 0) {
        return false;
    }

    int received = (high << 4) | low;

    return calculated == received;
}

static const char *nmea_field(const char *sentence, int field_index, int *field_length)
{
    const char *p = sentence;

    while (*p && *p != ',') {
        p++;
    }

    if (*p == '\0') {
        return NULL;
    }

    p++;

    for (int current = 0; current < field_index; current++) {
        while (*p && *p != ',' && *p != '*') {
            p++;
        }

        if (*p != ',') {
            return NULL;
        }

        p++;
    }

    const char *start = p;
    int length = 0;

    while (*p && *p != ',' && *p != '*' && *p != '\r' && *p != '\n') {
        p++;
        length++;
    }

    *field_length = length;
    return start;
}

static void copy_field(char *destination, int destination_size, const char *source, int source_length)
{
    int copy_length = source_length;

    if (copy_length > destination_size - 1) {
        copy_length = destination_size - 1;
    }

    memset(destination, 0, destination_size);
    memcpy(destination, source, copy_length);
    destination[copy_length] = '\0';
}

bool CookieGNSS_ParseEpoch(const char *buffer, uint32_t length, CookieGnssFix *fix)
{
    if (buffer == NULL || fix == NULL) {
        return false;
    }

    memset(fix, 0, sizeof(*fix));

    bool got_rmc = false;
    bool got_gga = false;
    bool got_gsa = false;

    const char *p = buffer;
    const char *end = buffer + length;

    while (p < end) {
        while (p < end && *p != '$') {
            p++;
        }

        if (p >= end) {
            break;
        }

        p++;  // Skip '$'

        const char *sentence_start = p;
        const char *line_end = p;

        while (line_end < end && *line_end != '\n') {
            line_end++;
        }

        int sentence_length = (int)(line_end - sentence_start);

        if (sentence_length < 6) {
            p = line_end + 1;
            continue;
        }

        if (!nmea_checksum_ok(sentence_start, sentence_length)) {
            p = line_end + 1;
            continue;
        }

        const char *type = sentence_start + 2;

        if (type[0] == 'R' && type[1] == 'M' && type[2] == 'C') {
            int field_length;
            const char *field;

            field = nmea_field(sentence_start, 0, &field_length);
            if (field && field_length > 0) {
                copy_field(fix->time_utc, COOKIE_GNSS_TIME_LEN, field, field_length);
            }

            field = nmea_field(sentence_start, 1, &field_length);
            if (field && field_length == 1) {
                fix->valid = (*field == 'A');
            }

            field = nmea_field(sentence_start, 6, &field_length);
            if (field && field_length > 0) {
                float speed_knots = strtof(field, NULL);
                fix->speed_cm_s = (uint16_t)(speed_knots / 1.94384f * 100.0f);
            }

            field = nmea_field(sentence_start, 7, &field_length);
            if (field && field_length > 0) {
                float cog_deg = strtof(field, NULL);
                fix->cog_cdeg = (uint16_t)(cog_deg * 100.0f);
            }

            field = nmea_field(sentence_start, 8, &field_length);
            if (field && field_length == 6) {
                copy_field(fix->date_ddmmyy, COOKIE_GNSS_DATE_LEN, field, field_length);
            }

            got_rmc = true;
        }

        else if (type[0] == 'G' && type[1] == 'G' && type[2] == 'A') {
            int field_length;
            const char *field;

            field = nmea_field(sentence_start, 1, &field_length);
            if (field && field_length > 0) {
                fix->latitude_raw = strtof(field, NULL);
            }

            field = nmea_field(sentence_start, 2, &field_length);
            if (field && field_length == 1) {
                fix->latitude_dir = *field;
            }

            field = nmea_field(sentence_start, 3, &field_length);
            if (field && field_length > 0) {
                fix->longitude_raw = strtof(field, NULL);
            }

            field = nmea_field(sentence_start, 4, &field_length);
            if (field && field_length == 1) {
                fix->longitude_dir = *field;
            }

            field = nmea_field(sentence_start, 8, &field_length);
            if (field && field_length > 0) {
                fix->altitude_m = strtof(field, NULL);
            }

            got_gga = true;
        }

        else if (type[0] == 'G' && type[1] == 'S' && type[2] == 'A') {
            int field_length;
            const char *field;

            field = nmea_field(sentence_start, 14, &field_length);
            if (field && field_length > 0) {
                float pdop = strtof(field, NULL);
                fix->pdop_x100 = (uint16_t)(pdop * 100.0f);
                got_gsa = true;
            }
        }

        p = line_end + 1;
    }

    return got_rmc && got_gga && got_gsa;
}