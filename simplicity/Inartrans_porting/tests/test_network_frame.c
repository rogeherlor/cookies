// gcc \
//   simplicity/Inartrans_porting/tests/test_network_frame.c \
//   simplicity/Inartrans_porting/src/network/network_frame.c \
//   -I simplicity/Inartrans_porting/src/network \
//   -o simplicity/Inartrans_porting/tests/test_network_frame

// ./simplicity/Inartrans_porting/tests/test_network_frame

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/network/network_frame.h"

static void read_u16(const uint8_t *buffer, size_t offset, uint16_t *value)
{
    memcpy(value, buffer + offset, sizeof(*value));
}

int main(void)
{
    uint8_t payload[COOKIE_NETWORK_DATA_PAYLOAD_SIZE] = {0};
    uint8_t message[COOKIE_NETWORK_DATA_FRAME_SIZE] = {0};

    for (uint8_t i = 0; i < COOKIE_NETWORK_DATA_PAYLOAD_SIZE; i++) {
        payload[i] = i;
    }

    CookieNetworkFrameHeader header = {0};

    header.packet_type = COOKIE_PACKET_TYPE_DATA;
    header.sender_rank = 1;
    header.destination = 0x0000;
    header.pan_id = 0x1234;
    header.source = 0xABCD;
    header.packet_number = 42;
    header.source_rank = 1;
    header.sequence = 7;

    bool built = CookieNetwork_BuildDataFrame(&header,
                                              payload,
                                              sizeof(payload),
                                              message,
                                              sizeof(message));

    assert(built);

    assert(message[0] == COOKIE_PACKET_TYPE_DATA);

    uint16_t value = 0;

    read_u16(message, 1, &value);
    assert(value == header.sender_rank);

    read_u16(message, 3, &value);
    assert(value == header.destination);

    read_u16(message, 5, &value);
    assert(value == header.pan_id);

    read_u16(message, 7, &value);
    assert(value == header.source);

    read_u16(message, 9, &value);
    assert(value == header.packet_number);

    read_u16(message, 11, &value);
    assert(value == header.source_rank);

    read_u16(message, 13, &value);
    assert(value == header.sequence);

    assert(message[15] == payload[0]);
    assert(message[89] == payload[74]);

    printf("Network frame test passed\n");
    printf("Header size: %u bytes\n", (unsigned)COOKIE_NETWORK_HEADER_SIZE);
    printf("Payload size: %u bytes\n", (unsigned)COOKIE_NETWORK_DATA_PAYLOAD_SIZE);
    printf("Frame size: %u bytes\n", (unsigned)COOKIE_NETWORK_DATA_FRAME_SIZE);
    printf("Packet type: %u\n", message[0]);
    printf("Payload starts at byte 15: %u\n", message[15]);
    printf("Payload ends at byte 89: %u\n", message[89]);

    return 0;
}