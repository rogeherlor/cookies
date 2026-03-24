/*
 * gps-uart.h
 *
 *  	CEI-UPM
 *      Author: Pablo Merino
 */

#ifndef GPS_UART_H_
#define GPS_UART_H_

/******  OJO AQUI ******/
#define GPS_UART_TX_LOC 30
#define GPS_UART_RX_LOC 28
/******  OJO AQUI ******/
#define MI_BUFFER_SIZE 1024

void gps_uart_set_input_handler(int (* handler)(unsigned char c));
int gps_putchar(int ch);
//void busca(void);
void set_GNSS_mode(uint8_t mode);
int parse_nmea_epoch(void);

extern USART_TypeDef * g_uart;
extern int (* input_handler)(unsigned char c);

#define PROBANDO 0

#ifndef COM_USART1_ENABLE		// martes
#define COM_USART1_ENABLE
#endif

extern uint8_t tiempo[10];
extern uint8_t tiemponuevo[10];
extern uint8_t fecha[6];
extern uint8_t tiempo2[10];
extern uint8_t validez;
extern uint8_t latitud[9];
extern uint8_t norte_sur;
extern uint8_t longitud[10];
extern uint8_t este_oeste;
extern uint8_t tamanoaltitud;
extern uint8_t altitud[8];
extern uint8_t metros;
extern uint8_t tengo_gps;

extern char mi_buffer[MI_BUFFER_SIZE];
extern uint32_t indice;


#endif /* GPS_UART_H_ */
