/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * main.c
 *
 * Code generation for function 'main'
 *
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
/* Include files */
#include "main.h"


/*
 * Definiciones importantes para el codigo:
 *
 * - NODO_QUE_ENVIA 1/0 : marca la diferencia entre nodo sensor y nodo coordinador.
 * - RED_USADA : la panID a la que se va a unir el nodo. Puede ser RED_PRUEBAS o RED_SCOTT1 (o cualquier otra que se quiera crear)
 * - SOLO_RELAY 1/0 si se utiliza solo como nodo router (0) o Sensor también (1)
 *
 *  Ambas definiciones se encuentran en protocolored.h
 *
 * - PROBANDO 1/0 : marca si el modulo GPS esta en modo normal (0), para su uso normal, o modo pruebas (1) con trama fija (incluso sin conectar el GPS)
 *  Se encuentra en gps-uart.h, normalmente fijo a 0 (no estamos en modo prueba, estamos funcionando normalmente)
 *
 * - TENGO_GPS 1/0 : para deshabilitar directamente el codigo del soporte para el GPS
 *  Se encuentra en flex-callbacks.c,  normalmente fijo a 1 (tengo/puede que tenga GPS, asi que habilito el soporte para ello)
 *
 * Para programar la placa, hay que darle al botón de Flash Programmer, y seleccionar el binario de la carpeta del proyecto (.hex)
 *
 *
 */


int MAIN(MAIN_FUNCTION_PARAMETERS)
{
  halInit();

#ifdef EMBER_AF_PLUGIN_MICRIUM_RTOS
  emberAfPluginMicriumRtosCpuInit();
#endif

  INTERRUPTS_ON();

  SERIAL_INIT();
  	  // Inicializamos el usart1 a 9600, que es el baudrate al que funciona el GPS
  emberSerialInit(COM_USART1, BAUD_9600, usartNoParity, 1);

  PRINT_RESET_INFORMATION();
  	  	  	  	  // Inicializamos los sensores termico e inercial
  emberAfPluginWstkSensorsInit();
  UTIL_init();
  GPIO_PinModeSet(gpioPortF, 11, gpioModePushPull, 1);	//inercial
  GPIO_PinOutSet(gpioPortF,11);		//habilito el enable del inercial
  IMU_init();
  IMU_config(20); //le damos la freq de muestreo del sensor, en Hz

  emberAfCorePrint("Probando hora: %d\n",  RTCDRV_GetWallClockTicks32()/4);	  // NOTA: esto da el tiempo desde el arranque, en ticks (4 ticks es un milisegundo)
  UTIL_delay(300);
  emberAfCorePrint("Probando hora otra vez: %d\n",  RTCDRV_GetWallClockTicks32()/4);
  #if defined(EMBER_AF_PLUGIN_FREE_RTOS)
  extern void emberPluginRtosInitAndRunConnectTask(void);
  emberPluginRtosInitAndRunConnectTask();
#elif defined(EMBER_AF_PLUGIN_MICRIUM_RTOS)
  emberAfPluginMicriumRtosInitAndRunTasks();
#else
  EmberStatus status;

  emberTaskEnableIdling(true);

  emAppTask = emberTaskInit(emAppEvents);		//Tareas periodicas

  // Initialize the radio and the stack.  If this fails, we have to assert
  // because something is wrong.
  status = emberInit();
  emberSerialPrintfLine(APP_SERIAL, "Init: 0x%x", status);
  assert(status == EMBER_SUCCESS);

  emberAfInit();
  emberAfMainInitCallback();
  	  	  	  // Inicializamos a 0 todas las var. auxiliares que usamos para procesar la trama del GPS
	norte_sur=0;
	este_oeste=0;
	metros=0;
	for (int i=0;i<10;i++){
		if (i<sizeof(tiempo2)) { tiempo2[i]=0;	}
		if (i<sizeof(tiempo)) { tiempo[i]=0;	}
		if (i<sizeof(latitud)) { latitud[i]=0;	}
		if (i<sizeof(longitud)) { longitud[i]=0;	}
		if (i<sizeof(altitud)) { altitud[i]='0';	}
	}
		// Habilitamos las interrupciones de entrada de la uart del gps
	NVIC_EnableIRQ (USART1_RX_IRQn);
    USART_Enable(USART1, usartEnable);
    USART_IntEnable(USART1, USART_IF_TXBL | USART_IF_RXDATAV);
    USART_IntEnable(USART1,USART_IEN_RXDATAV);
    	// Arrancamos el stack de red, dependiendo de si es nodo sensor o coordinador
    arrancar_red();


#if(RED_USADA==RED_JAIME)			// nota para envios simples: longitud_simple es la longitud del paquete enviado. retocar en funcion de lo que se envia
	uint8_t prueba[50] = {0};
	memcpy(prueba, "blablabla", strlen("blablabla"));
	longitud_simple = sizeof(prueba);
	emberAfCorePrint("\nLongitud del envio: %u\n", longitud_simple);
#endif
#if NODO_QUE_ENVIA											// el nodo que recibe NO tiene el report periodico
#if (RED_USADA != RED_JAIME)				// para la prueba de envios de seguridad no queremos reports extraños, simplemente el envio
#if (SOLO_RELAY != 1)
    	emberEventControlSetDelayMS(reportControl, 1000);		// tiempo inicial hasta el primer envio periodico, en ms
#else
    	UTIL_delay(500);
    	//envio_pruebas(prueba, 0x0000);
#endif		// del solo_relay
#endif
#endif


  while (true) {
    halResetWatchdog();

    // Let the stack or EZSP layer run periodic tasks.
    emberTick();

    // Let the application and plugins run periodic tasks.
    emberAfMainTickCallback();
    emberAfTick();

    emberRunTask(emAppTask);			//Tareas periodicas

  }
#endif

  return 0;
}
