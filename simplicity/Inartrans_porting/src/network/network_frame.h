#ifndef COOKIE_NETWORK_FRAME_H
#define COOKIE_NETWORK_FRAME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define COOKIE_NETWORK_HEADER_SIZE 15U
#define COOKIE_NETWORK_DATA_PAYLOAD_SIZE 75U
#define COOKIE_NETWORK_DATA_FRAME_SIZE \
    (COOKIE_NETWORK_HEADER_SIZE + COOKIE_NETWORK_DATA_PAYLOAD_SIZE)

typedef enum {
    COOKIE_PACKET_TYPE_DISCOVERY = 1,
    COOKIE_PACKET_TYPE_CONFIRMATION = 2,
    COOKIE_PACKET_TYPE_DATA = 3,
    COOKIE_PACKET_TYPE_REPAIR_BROADCAST = 4,
    COOKIE_PACKET_TYPE_REPAIR_UNICAST = 5,
    COOKIE_PACKET_TYPE_REQUEST = 6,
    COOKIE_PACKET_TYPE_SIMPLE = 7,
    COOKIE_PACKET_TYPE_CONFIG = 8
} CookiePacketType;

typedef struct {
    uint8_t packet_type;

    uint16_t sender_rank;
    uint16_t destination;
    uint16_t pan_id;
    uint16_t source;

    uint16_t packet_number;
    uint16_t source_rank;
    uint16_t sequence;
} CookieNetworkFrameHeader;

bool CookieNetwork_BuildDataFrame(const CookieNetworkFrameHeader *header,
                                  const uint8_t *payload,
                                  size_t payload_size,
                                  uint8_t *message,
                                  size_t message_size);

#endif /* COOKIE_NETWORK_FRAME_H */