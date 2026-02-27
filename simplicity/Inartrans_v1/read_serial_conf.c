/*
 * read_serial_conf.c
 *
 *  Created on: 7 de jul. de 2022
 *      Author: PCNET22Win10 Rogelio
 */

#include "read_serial_conf.h"

#include "serial/serial.h"

typedef uint8_t EmberStatus;

uint8_t port = 0x20;
uint8_t dataByte;
uint16_t length;
uint16_t bytesRead;

uint16_t read_bytes_avaible;
EmberStatus status;
uint16_t aux;

char data[10];
uint8_t max = 10;

//////////Al final no se usa, se introduce por comandos declarados en flex-cli.c

void read_serial_conf(void){

	read_bytes_avaible = emberSerialReadAvailable(port); 						//Se lee puerto ttyUSB0=0x20
	//status = emberSerialPrintf(0x20, "Test bytes avaible %u \n", read_bytes_avaible);		//Test print funciona

	if (read_bytes_avaible > 0){
		status = emberSerialReadLine(port, &data, max);
		status = emberSerialPrintf(0x20, "Test read data %s \n", data);
	}

}


