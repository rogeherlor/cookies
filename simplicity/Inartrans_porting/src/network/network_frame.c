#include "network_frame.h"

#include <string.h>

/*
 * Legacy Inartrans_v2 network frame layout.
 *
 * The network frame wraps the 75-byte data packet with a 15-byte header:
 *
 *   bytes 0..14   -> network header
 *   bytes 15..89  -> data payload
 *
 * This module does not send anything through the radio. It only builds the
 * binary frame that will later be passed to the network/radio layer.
 */

#define OFF_PACKET_TYPE     0U
#define OFF_SENDER_RANK     1U
#define OFF_DESTINATION     3U
#define OFF_PAN_ID          5U
#define OFF_SOURCE          7U
#define OFF_PACKET_NUMBER   9U
#define OFF_SOURCE_RANK     11U
#define OFF_SEQUENCE        13U
#define OFF_PAYLOAD         COOKIE_NETWORK_HEADER_SIZE

static void write_bytes(uint8_t *buffer, size_t offset,
                        const void *src, size_t size)
{
    memcpy(buffer + offset, src, size);
}

bool CookieNetwork_BuildDataFrame(const CookieNetworkFrameHeader *header,
                                  const uint8_t *payload,
                                  size_t payload_size,
                                  uint8_t *message,
                                  size_t message_size)
{
    if (header == NULL || payload == NULL || message == NULL) {
        return false;
    }

    if (payload_size != COOKIE_NETWORK_DATA_PAYLOAD_SIZE) {
        return false;
    }

    if (message_size < COOKIE_NETWORK_DATA_FRAME_SIZE) {
        return false;
    }

    memset(message, 0, COOKIE_NETWORK_DATA_FRAME_SIZE);

    write_bytes(message, OFF_PACKET_TYPE,
                &header->packet_type,
                sizeof(header->packet_type));

    write_bytes(message, OFF_SENDER_RANK,
                &header->sender_rank,
                sizeof(header->sender_rank));

    write_bytes(message, OFF_DESTINATION,
                &header->destination,
                sizeof(header->destination));

    write_bytes(message, OFF_PAN_ID,
                &header->pan_id,
                sizeof(header->pan_id));

    write_bytes(message, OFF_SOURCE,
                &header->source,
                sizeof(header->source));

    write_bytes(message, OFF_PACKET_NUMBER,
                &header->packet_number,
                sizeof(header->packet_number));

    write_bytes(message, OFF_SOURCE_RANK,
                &header->source_rank,
                sizeof(header->source_rank));

    write_bytes(message, OFF_SEQUENCE,
                &header->sequence,
                sizeof(header->sequence));

    write_bytes(message, OFF_PAYLOAD, payload, payload_size);

    return true;
}