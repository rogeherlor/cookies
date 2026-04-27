// To compile and run this test, use the following command from the project root:
// gcc \
//   simplicity/Inartrans_porting/tests/test_gnss_parser.c \
//   simplicity/Inartrans_porting/src/gnss/gnss.c \
//   -I simplicity/Inartrans_porting/src/gnss \
//   -o simplicity/Inartrans_porting/tests/test_gnss_parser

// Then execute the test binary:
// ./simplicity/Inartrans_porting/tests/test_gnss_parser

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "../src/gnss/gnss.h"

int main(void)
{
    const char *nmea_epoch =
        "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A*62\r\n"
        "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59\r\n"
        "$GNGSA,A,3,04,05,09,12,24,25,29,31,,,,,1.8,1.0,1.5*23\r\n";

    CookieGnssFix fix;

    bool ok = CookieGNSS_ParseEpoch(nmea_epoch, strlen(nmea_epoch), &fix);

    if (!ok) {
        printf("GNSS epoch parsing failed\n");
        return 1;
    }

    printf("GNSS epoch parsed successfully\n");
    printf("Valid: %s\n", fix.valid ? "true" : "false");
    printf("Time UTC: %s\n", fix.time_utc);
    printf("Date DDMMYY: %s\n", fix.date_ddmmyy);
    printf("Latitude raw: %.3f %c\n", fix.latitude_raw, fix.latitude_dir);
    printf("Longitude raw: %.3f %c\n", fix.longitude_raw, fix.longitude_dir);
    printf("Altitude: %.2f m\n", fix.altitude_m);
    printf("Speed: %u cm/s\n", fix.speed_cm_s);
    printf("COG: %u cdeg\n", fix.cog_cdeg);
    printf("PDOP: %u\n", fix.pdop_x100);

    return 0;
}
