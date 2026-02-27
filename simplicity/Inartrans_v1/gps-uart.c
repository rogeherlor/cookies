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
uint8_t tiempo_ultimo_valido[10] = "000000.000";
uint8_t tiemponuevo[10] = {0};
uint8_t fechanueva[6] = {0};
uint8_t vel_GNSS[6] = {0}; 	//Feb23
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
gps_uart_init(void)				// funcion innecesaria, no se usa. Inicializamos directametne en el main con mucho menos
{
  USART_InitAsync_TypeDef init  = USART_INITASYNC_DEFAULT;

#ifdef SERIAL_BAUDRATE
  init.baudrate = SERIAL_BAUDRATE;
#endif /* SERIAL_BAUDRATE */

  CMU_ClockEnable(cmuClock_HFPER, true);
  CMU_ClockEnable(cmuClock_GPIO, true);
  CMU_ClockEnable(cmuClock_USART1, true);

  GPIO_PinModeSet(GPS_TX_PORT, GPS_TX_PIN, gpioModePushPull, 1);

#ifdef GPS_UART_RX_LOC
  GPIO_PinModeSet(GPS_RX_PORT, GPS_RX_PIN, gpioModeInput, 0);
#endif /* GPS_UART_RX_LOC */

  init.enable = usartEnableRx;
  init.baudrate = BAUD_9600; // Por si queremos cambiar aqui el baudrate (por defecto a 115200)

  USART_InitAsync(g_uart, &init);

  /* Raise TXBL as soon as there is at least one empty slot */
  g_uart->CTRL |= USART_CTRL_TXBIL;

#ifdef GPS_UART_RX_LOC
  g_uart->ROUTELOC0 = (g_uart->ROUTELOC0 & ~(_USART_ROUTELOC0_TXLOC_MASK
                                            | _USART_ROUTELOC0_RXLOC_MASK ))
                       | (GPS_TX_LOC << _USART_ROUTELOC0_TXLOC_SHIFT)
                       | (GPS_RX_LOC << _USART_ROUTELOC0_RXLOC_SHIFT);
  g_uart->ROUTEPEN |= USART_ROUTEPEN_RXPEN | USART_ROUTEPEN_TXPEN;
#else /* GPS_UART_RX_LOC */
  g_uart->ROUTELOC0 = (g_uart->ROUTELOC0 & ~(_USART_ROUTELOC0_TXLOC_MASK
                                            | _USART_ROUTELOC0_RXLOC_MASK ))
                       | (GPS_TX_LOC << _USART_ROUTELOC0_TXLOC_SHIFT);
  g_uart->ROUTEPEN |= USART_ROUTEPEN_TXPEN;
#endif /* GPS_UART_RX_LOC */

  USART_Enable(g_uart, usartEnable);

  USART_IntEnable(g_uart, USART_IF_TXBL | USART_IF_RXDATAV);
  /* USART0_TX_IRQn will be enabled in NVIC when there is data to send */
}
/*---------------------------------------------------------------------------*/
int
gps_inputbyte(unsigned char c)		// funcion innecesaria, no se usa. Procesamos el input directamente en USART1_RX_IRQHandler (com.c)
{
	if (indice<MI_BUFFER_SIZE){
		memcpy(mi_buffer+indice,&c,sizeof(c));
		indice++;
	}
		return 1;
}
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

int busca2 (void){

	int iniciot = 0;
	int faseuno = 0;
	int fasedos = 0;
	int fasetres = 0;
	int flag_tiempo_nuevo = 0;
	uint16_t comas[14]={0}; 		//Feb23 Valor original 14 comas. Si parse mal y coge mas comas por error desborda y se resetea-
	uint16_t comasfecha[16]={0};	//Feb23 12
	for (int i=0;i<14;i++){
		comas[i]=0;
		if(i<12){
			comasfecha[i]=0;
		}
	}

	uint8_t lacoma = 0;
	uint8_t flag_fecha = 0;
/*---------------------------------------------------------------------------*/
#if PROBANDO==0
#if VERBOSEO
	emberAfCorePrint("Aqui viene el buffer: \n");
	for (int i=0; i < indice;i++){
		emberAfCorePrint("%c",mi_buffer[i]);
	}
	emberAfCorePrint("\n");
#endif
	for (int i=0; i < indice;i++){

		if ((mi_buffer[i+1]==0x47)&&(mi_buffer[i+3]==0x52)&&(mi_buffer[i+4]==0x4d)&&(mi_buffer[i+5]==0x43)){ //Ene23 Hex to char 0x47=G, 0x52=R, 0x4d=M, 0x43=C
			iniciot = i+7;
			memcpy(&tiemponuevo, &mi_buffer[iniciot],10);
			memcpy(&validez, &mi_buffer[iniciot+11],1);
			faseuno = 1;							// hace que se lea en orden. Primero siempre un RMC
			for (int j=i+5;j<indice;j++){			// aqui busco la posicion de las comas del mensaje GPRMC, donde encontrare la fecha
				if (mi_buffer[j]==0x2c){
					comasfecha[flag_fecha] = j;
					flag_fecha++;
					if (flag_fecha==11){
						memcpy(&vel_GNSS, "000000", 6);
						memcpy(&vel_GNSS, &mi_buffer[comasfecha[6]+1], comasfecha[7] - comasfecha[6] - 1);
						memcpy(&fechanueva, &mi_buffer[comasfecha[8]+1], comasfecha[9] - comasfecha[8] - 1);
						flag_fecha=0;
						break;
					}
				}
			}
		}	//fin if mensaje $GPRMC

		if ((mi_buffer[i+1]==0x47)&&(mi_buffer[i+3]==0x47)&&(mi_buffer[i+4]==0x47)&&(mi_buffer[i+5]==0x41)&&(faseuno==1)){ //Ene23 Hex to char 0x47=G, 0x41=A
			for (int j=i+5;j<indice;j++){
				if (mi_buffer[j]==0x2c){	//Ene23 Hex to char 0x2c=,
					comas[lacoma] = j;
					lacoma++;
					if (lacoma==14){
						fasedos = 1;
						break;
					}
				}
			}
			/* Feb23 A�adida fasetres por lo que no se puede salir aqui. Supongo que esto era para no leer el siguiente RMC por error, pero se muestrea tan rapido que no hace falta, no se solapan varios mensajes nunca.
			if (lacoma==14){
				fasedos = 1;
				break; //Feb23 Este break obliga a que no se lea el $--RMC siguiente
			}
			*/

		}//fin if mensaje $GPGGA. //Feb23 En GNSS parece que el valor de long y lat en GPGGA es el mismo que en GNRMC, por lo que GPGGA parece que no lo saca del GP sino de la mezcla GN.

		if ((mi_buffer[i]==0x24)&&(mi_buffer[i+3]==0x47)&&(mi_buffer[i+4]==0x53)&&(mi_buffer[i+5]==0x41)&&(fasedos==1)){ //Feb23 $--GSA para el PDOP; Hex to char 0X24=$, 0x47=G, 0x53=S, 0x41=A
			for (int j=i+5;j<indice;j++){			// aqui busco la posicion de las comas del mensaje
				if (mi_buffer[j]==0x2c){
					comasfecha[flag_fecha] = j;
					flag_fecha++;
					if (flag_fecha==17){
						if((comasfecha[15]-comasfecha[14])>1) { 	//Si hay valor de PDOP se guarda, si no, esta vacio y se limpia el buffer
							memcpy(&PDOP, "000000", 6);
							memcpy(&PDOP, &mi_buffer[comasfecha[14]+1], comasfecha[15] - comasfecha[14] - 1);
							fasetres = 1;
						}else{
							for (int i=0;i<sizeof(mi_buffer);i++){mi_buffer[i]='0';}
							indice = 0;
						}
						flag_fecha=0;
						break;
					}
				}
			}
		}	//fin if mensaje $--GSA. Ver si crear fasetres para asegurar que se guarda

	}
	for (int i=0;i<lacoma;i++){	//compruebo que todas las comas hayan sido asignadas y sean posiciones distintas, o FALLO
		if (comas[i] < 5){
			validez = 0x56;		//Ene23 Hex to char 0x56=V
		}
		for (int j=i+1;j<lacoma;j++){
			if (comas[j] == comas[i]){
				validez = 0x56;
			}
		}
	}	//fin de la comprobacion de mensaje valido
#if VERBOSEO
	emberAfCorePrint("\nIndice: %u\n",indice);
	emberAfCorePrint("Comas:");
	for (int i=0;i<lacoma;i++){
		emberAfCorePrint(" %u",comas[i]);
	}
	emberAfCorePrint("\n");
#endif
//	if (indice < 160){
//		validez = 0x56;
//	}
	if (fasetres == 1){
		memcpy(&tiempo, &tiemponuevo, 10);
		memcpy(&fecha, &fechanueva, 6);
		if((fecha[4]=='8')&&(fecha[5]=='0')){
			memcpy(&fecha, "000000", 6);
			memcpy(&tiempo, "000000.000", 10);
			GPIO_PinOutSet(gpioPortA, 6);
		} else {
			flag_tiempo_nuevo = 1;
			GPIO_PinOutClear(gpioPortA, 6);
		}
		mi_timeout = 0;
	} else{
		mi_timeout++;
		if (mi_timeout>=10){			// timeout en 2 segundos (tras X tiempo sin procesar dato nuevo pasa a indicar trama invalida)
			validez = 0x56;
		}
	}
#endif
#if PROBANDO
		memset(&mi_buffer,0,MI_BUFFER_SIZE);
		indice = 0;

		tamanoaltitud = strlen("573.7122");
		memcpy(&validez, "A",1);
		memcpy(&tiempo,"133334.000",strlen("133334.000"));
		memcpy(&tiempo2,"133534.000",strlen("133534.000"));
		memcpy(&latitud,"4026.4023",strlen("4026.4023"));
		memcpy(&norte_sur, "N", sizeof(char));
		memcpy(&longitud, "00341.3207", strlen("00341.3207"));
		memcpy(&este_oeste, "W", sizeof(char));
		memcpy(&altitud, "00000000", sizeof(altitud));
		memcpy(&altitud, "573.7122", tamanoaltitud);
		memcpy(&metros, "M", sizeof(char));
		flag_tiempo_nuevo = 1;
#else
		if (fasetres==1){
	if (validez == 0x41){	// validez es A, lectura de satelites correcta
		//GPIO_PinOutSet(gpioPortA,5);
		for (int i=0;i<sizeof(altitud);i++) {altitud[i]='0';}
		tamanoaltitud = comas[9]-comas[8]-1;
			if (tamanoaltitud > sizeof(altitud)){
				tamanoaltitud = sizeof(altitud);
			}
		memcpy(&tiempo2, &mi_buffer[comas[0]+1],sizeof(tiempo2));//comas[1]-comas[0]-1);
		memcpy(&latitud, &mi_buffer[comas[1]+1],sizeof(latitud));//comas[2]-comas[1]-1);
		memcpy(&norte_sur, &mi_buffer[comas[2]+1],sizeof(norte_sur));//comas[3]-comas[2]-1);
		memcpy(&longitud, &mi_buffer[comas[3]+1],sizeof(longitud));//comas[4]-comas[3]-1);
		memcpy(&este_oeste, &mi_buffer[comas[4]+1],sizeof(este_oeste));//comas[5]-comas[4]-1);
		memcpy(&altitud, &mi_buffer[comas[8]+1],tamanoaltitud);//sizeof(altitud));*/comas[9]-comas[8]-1);
		memcpy(&metros,"M",sizeof(char));
		// la longitud de todas esas cosas es fija y conocida, menos la altitud que puede cambiar como quiera, necesito
		// parsear la altitud por las comas, coma final - coma de inicio - 1
	} else {			// validez es V, lectura de satelites no valida
		memcpy(&tiempo2, "XinvalidoX",sizeof(tiempo2));
		memcpy(&latitud, "XinvalidX",sizeof(latitud));
		memcpy(&norte_sur, "X",sizeof(norte_sur));
		memcpy(&longitud, "XinvalidoX",sizeof(longitud));
		memcpy(&este_oeste, "X",sizeof(este_oeste));
		memcpy(&altitud, "Xinvalid",sizeof(altitud));
		memcpy(&metros, "Y",sizeof(metros));
	}
		}		// fin del if fasetres==1
#endif
	// comprobamos que lo que estamos enviando no son caracteres basura
	for (int i=0;i<sizeof(tiempo2);i++){
		if (tiempo2[i]==0x2c){		//Ene23 Hex to char 0x2c=, 0x56=V
			validez = 0x56;
		}
	}
	for (int i=0;i<sizeof(latitud);i++){
		if (latitud[i]==0x2c){
			validez = 0x56;
		}
	}
	for (int i=0;i<sizeof(longitud);i++){
		if (longitud[i]==0x2c){
			validez = 0x56;
		}
	}
	for (int i=0;i<sizeof(altitud);i++){
		if (altitud[i]==0x2c){
			validez = 0x56;
		}
	}

	// al terminar, limpio el buffer y pongo el indice a 0
	if (fasetres==1){
		for (int i=0;i<sizeof(mi_buffer);i++){
			mi_buffer[i]='0';
		}
		indice = 0;
	}
	//return (validez==0x41); //Ene23 Hex to char 0x41=A
	return flag_tiempo_nuevo;
}	// fin de busca2
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
