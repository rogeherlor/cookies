/*
 * gps-uart.c
 *
 *  	CEI-UPM
 *      Author: Pablo Merino
 */


#include <em_device.h>
#include <em_usart.h>
#include <em_cmu.h>
#include <em_gpio.h>
#include "gps-uart.h"
//#include "lib/dbg-io/dbg.h"
//#include "debug-uart.h"
#include <stdbool.h>
#include <stdio.h>
//#include "lib/ringbuf.h"
#include <string.h>
#ifndef PGM
#define PGM     const
#endif
#include "serial.h"
#include "protocolored.h"
#include "hal/plugin/serial/com.h"
/***************************************** OJO *************/

#ifndef GPS_UART_TX_LOC
#error "gps-uart: no UART TX_LOC configured"
#endif /* GPS_UART_TX_LOC */

#define GPS_TX_LOC        GPS_UART_TX_LOC
#define GPS_TX_PORT       AF_USART1_TX_PORT(GPS_TX_LOC)
#define GPS_TX_PIN        AF_USART1_TX_PIN(GPS_TX_LOC)

#ifdef GPS_UART_RX_LOC
#define GPS_RX_LOC        GPS_UART_RX_LOC
#define GPS_RX_PORT       AF_USART1_RX_PORT(GPS_RX_LOC)
#define GPS_RX_PIN        AF_USART1_RX_PIN(GPS_RX_LOC)
#endif /* GPS_UART_RX_LOC */

#ifdef GPS_UART_CONF_TX_BUFFER
#define GPS_TX_SIZE GPS_UART_CONF_TX_BUFFER
#else /* GPS_UART_CONF_TX_BUFFER */
#define GPS_TX_SIZE       128
#endif /* GPS_UART_CONF_TX_BUFFER */


#define HANDLED_RX_ERR (USART_IF_FERR | USART_IF_PERR | USART_IF_RXOF)

static uint8_t TXBUFFER[GPS_TX_SIZE] = {0};
/* first char to read */
static volatile uint16_t rpos = 0;
/* last char written (or next to write) */
static volatile uint16_t wpos = 0;
char mi_buffer[MI_BUFFER_SIZE]= {0}; //Feb22 En modo GPS only un mensaje alrededor de 540 bytes. En modo GNSS con GP,GN,GA alrededor de 680. Guarda el mensaje hasta que tiene un RMC y un GGA, luego borra el buffer
uint32_t indice = 0;
uint8_t validez = 0x56;
uint8_t tiempo[10] = "000000.000";
uint8_t tiemponuevo[10] = {0};
uint8_t fechanueva[6] = {0};
uint8_t vel_GNSS[6] = {0}; 	//Feb23
uint8_t cog_GNSS[6] = {0};
uint8_t PDOP[6] = {0};		//Feb23
uint8_t fecha[6] = {0};
uint8_t tiempo2[10] = {0};
uint8_t latitud[9] = {0};
uint8_t norte_sur = 0;
uint8_t longitud[10] = {0};
uint8_t este_oeste = 0;
uint8_t tamanoaltitud = 0;
uint8_t altitud[8] = {0};
uint8_t metros = 0;
uint8_t tengo_gps = 0;
uint8_t mi_timeout = 0;

//void USART1_RX_IRQHandler() __attribute__((interrupt));
//void USART1_TX_IRQHandler() __attribute__((interrupt));

 USART_TypeDef * g_uart = USART1;
 int (* input_handler)(unsigned char c);


/*---------------------------------------------------------------------------*/
static void
send_txbuf(void)
{
  while(rpos != wpos && (g_uart->STATUS & USART_STATUS_TXBL)) {
    g_uart->TXDATA = (uint32_t)TXBUFFER[rpos];
    rpos = ((rpos + 1) % GPS_TX_SIZE);
  }
}
/*---------------------------------------------------------------------------*/
static inline void
write_txbuf(char c)
{
  TXBUFFER[wpos] = c;
  wpos = ((wpos + 1) % GPS_TX_SIZE);
}
/*---------------------------------------------------------------------------*/
static inline bool
is_tx_buffer_full(void)
{
  return wpos + 1 == rpos || (wpos + 1 == GPS_TX_SIZE && rpos == 0);
}
/*---------------------------------------------------------------------------*/
static inline bool
clear_full_fifo(void)
{
  if(is_tx_buffer_full()) {
    rpos = ((rpos + 1) % GPS_TX_SIZE);
    return true;
  } else {
    return false;
  }
}
/*---------------------------------------------------------------------------*/
static inline void
write_byte(char c)						// funcion innecesaria, no se usa. Nunca vamos a ESCRIBIR nada al GPS
{
  static bool dropping = false;
  if(is_tx_buffer_full()) {
    if(!dropping) {
      /* Wait for a short time to allow current data byte to finish transfer */
    }
  }

  NVIC_DisableIRQ(USART1_TX_IRQn);
  dropping = clear_full_fifo();
  write_txbuf(c);
  send_txbuf();
  if (rpos != wpos) {
    NVIC_ClearPendingIRQ(USART1_TX_IRQn);
    NVIC_EnableIRQ(USART1_TX_IRQn);
  }
}
/*---------------------------------------------------------------------------*/

int
gps_uart_is_busy(void)
{
  return rpos != wpos;
}
/*---------------------------------------------------------------------------*/
void
gps_uart_set_input_handler(int (* handler)(unsigned char c))
{

  input_handler = handler;

  USART_IntClear(g_uart, HANDLED_RX_ERR);
  NVIC_ClearPendingIRQ(USART1_RX_IRQn);
  USART_IntEnable(g_uart, USART_IF_RXDATAV | HANDLED_RX_ERR);
  NVIC_EnableIRQ(USART1_RX_IRQn);
}
/*---------------------------------------------------------------------------*/
unsigned int
gps_send_bytes(const unsigned char *seq, unsigned int len)
{
  for(int i=0; i < len; i++) {
	  gps_putchar(seq[i]);
  }

  return len;
}
/*---------------------------------------------------------------------------*/
int
gps_putchar(int ch)
{
  write_byte(ch);
  return ch;
}

void set_GNSS_mode(uint8_t mode){	//Feb23 Los modos van de peores a mejore para que se pueda iterar a partir de cierto modo sin pasar por los anteriores
	if(mode==1){
		COM_PrintfLine(COM_USART1,"$PMTK301,0*2C");						//Set no DGPS mode
		COM_PrintfLine(COM_USART1,"$PMTK353,1,0,0,0*36");				//Set GPS mode, ACK = "$PMTK001,..."
	}
	if(mode==2) COM_PrintfLine(COM_USART1,"$PMTK353,0,1,0,0*36"); 		//Set GLONASS mode, ACK = "$PMTK001,..."
	if(mode==3) COM_PrintfLine(COM_USART1,"$PMTK353,0,0,1,0*36"); 		//Set Galileo, ACK = "$PMTK001,..."
	if(mode==4){
		COM_PrintfLine(COM_USART1,"$PMTK301,2*2E");						//Set DGPS mode
		COM_PrintfLine(COM_USART1,"$PMTK353,1,0,0,0*36");				//Set GPS mode, ACK = "$PMTK001,..."
	}
	if(mode==5){
		COM_PrintfLine(COM_USART1,"$PMTK301,0*2C");						//Set no DGPS mode
		COM_PrintfLine(COM_USART1,"$PMTK353,1,1,0,0*37"); 				//Set GPS + GLONASS, ACK = "$PMTK001,..."
	}
	if(mode==6){
		COM_PrintfLine(COM_USART1,"$PMTK301,0*2C");						//Set no DGPS mode
		COM_PrintfLine(COM_USART1,"$PMTK353,1,1,1,0*36"); 				//Set GPS + GLONASS + Galileo, ACK = "$PMTK001,..."
	}
	if(mode==7) {
		COM_PrintfLine(COM_USART1,"$PMTK301,2*2E");						//Set DGPS mode
		COM_PrintfLine(COM_USART1,"$PMTK353,1,1,1,0*36"); 				//Set GPS + GLONASS + Galileo, ACK = "$PMTK001,..."
	}

	for (int i=0;i<sizeof(mi_buffer);i++){								//Limpiar buffer del modo anterior
		mi_buffer[i]='0';
	}
	indice = 0;
	validez = 0x56;
}

/* Validates the NMEA *XX checksum.
 * s points to the first byte after '$'; len is the number of bytes up to
 * and including the '\n' (or end of buffer).
 * Returns 1 if the checksum matches, 0 otherwise. */
static int nmea_checksum_ok(const char *s, int len)
{
  int i, calc = 0;
  for (i = 0; i < len && s[i] != '*'; i++)
    calc ^= (unsigned char)s[i];
  if (i + 2 >= len) return 0;
  int hi = s[i + 1], lo = s[i + 2];
#define HEX(c) ((c)>='0'&&(c)<='9' ? (c)-'0' : \
                (c)>='A'&&(c)<='F' ? (c)-'A'+10 : \
                (c)>='a'&&(c)<='f' ? (c)-'a'+10 : -1)
  int hi_v = HEX(hi), lo_v = HEX(lo);
#undef HEX
  if (hi_v < 0 || lo_v < 0) return 0;
  return calc == ((hi_v << 4) | lo_v);
}

/* Returns a pointer to field n (0-indexed after the sentence type) and
 * writes its length to *field_len.  Returns NULL if the field is absent. */
static const char *nmea_field(const char *sentence, int n, int *field_len)
{
  const char *p = sentence;
  while (*p && *p != ',') p++;
  if (!*p) return NULL;
  p++;
  int f = 0;
  while (f < n) {
    while (*p && *p != ',' && *p != '*') p++;
    if (*p != ',') return NULL;
    p++; f++;
  }
  const char *start = p;
  int len = 0;
  while (*p && *p != ',' && *p != '*' && *p != '\r' && *p != '\n') { p++; len++; }
  *field_len = len;
  return start;
}

/* Replaces busca2(): single O(n) forward pass with checksum validation and
 * order-independent sentence detection (RMC / GGA / GSA in any order).
 * Returns 1 when a complete, valid epoch has been parsed and committed to the
 * global arrays (tiempo, fecha, latitud, …), 0 otherwise. */
int parse_nmea_epoch(void)
{
  uint8_t got = 0;           /* bitmask: bit0=RMC, bit1=GGA, bit2=GSA */
  int flag_tiempo_nuevo = 0;
  const char *p   = mi_buffer;
  const char *end = mi_buffer + indice;

  while (p < end) {
    /* find next sentence start */
    while (p < end && *p != '$') p++;
    if (p >= end) break;
    p++;   /* skip '$' */

    const char *sent_start = p;
    const char *nl = p;
    while (nl < end && *nl != '\n') nl++;
    int sent_len = (int)(nl - sent_start);
    if (sent_len < 6) { p = nl + 1; continue; }

    if (!nmea_checksum_ok(sent_start, sent_len)) { p = nl + 1; continue; }

    /* sent_start[0..1] = talker (GP/GN/GL…), [2..4] = sentence type */
    const char *type = sent_start + 2;

    if (type[0]=='R' && type[1]=='M' && type[2]=='C') {
      int fl; const char *f;
      f = nmea_field(sent_start, 0, &fl);
      if (f && fl > 0) memcpy(tiemponuevo, f, fl < 10 ? fl : 10);
      f = nmea_field(sent_start, 1, &fl);
      if (f && fl == 1) validez = (uint8_t)*f;
      f = nmea_field(sent_start, 6, &fl);
      if (f) { memcpy(vel_GNSS, "000000", 6); memcpy(vel_GNSS, f, fl < 6 ? fl : 6); }
      f = nmea_field(sent_start, 7, &fl);
      if (f) { memcpy(cog_GNSS, "000000", 6); memcpy(cog_GNSS, f, fl < 6 ? fl : 6); }
      f = nmea_field(sent_start, 8, &fl);
      if (f && fl == 6) {
        memcpy(fechanueva, f, 6);
        if (!(fechanueva[4] == '8' && fechanueva[5] == '0'))
          flag_tiempo_nuevo = 1;
      }
      got |= 0x01;

    } else if (type[0]=='G' && type[1]=='G' && type[2]=='A') {
      int fl; const char *f;
      f = nmea_field(sent_start, 0, &fl);
      if (f && fl > 0) memcpy(tiempo2, f, fl < 10 ? fl : 10);
      f = nmea_field(sent_start, 1, &fl);
      if (f && fl > 0) memcpy(latitud,  f, fl < 9  ? fl : 9);
      f = nmea_field(sent_start, 2, &fl);
      if (f && fl == 1) norte_sur = (uint8_t)*f;
      f = nmea_field(sent_start, 3, &fl);
      if (f && fl > 0) memcpy(longitud, f, fl < 10 ? fl : 10);
      f = nmea_field(sent_start, 4, &fl);
      if (f && fl == 1) este_oeste = (uint8_t)*f;
      f = nmea_field(sent_start, 8, &fl);
      if (f && fl > 0) {
        tamanoaltitud = fl < 8 ? fl : 8;
        for (int i = 0; i < 8; i++) altitud[i] = '0';
        memcpy(altitud, f, tamanoaltitud);
      }
      got |= 0x02;

    } else if (type[0]=='G' && type[1]=='S' && type[2]=='A') {
      int fl; const char *f;
      /* field 14 = PDOP; skip GSA sentences with empty PDOP field */
      f = nmea_field(sent_start, 14, &fl);
      if (f && fl > 0) {
        memcpy(PDOP, "000000", 6);
        memcpy(PDOP, f, fl < 6 ? fl : 6);
        got |= 0x04;
      }
    }

    p = nl + 1;
  }

  if (got == 0x07 && flag_tiempo_nuevo) {
    memcpy(tiempo, tiemponuevo, 10);
    memcpy(fecha,  fechanueva,  6);
    GPIO_PinOutClear(gpioPortA, 6);
    mi_timeout = 0;
    for (int i = 0; i < MI_BUFFER_SIZE; i++) mi_buffer[i] = '0';
    indice = 0;
    return 1;
  }

  mi_timeout++;
  if (mi_timeout >= 10) validez = 0x56;
  return 0;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
