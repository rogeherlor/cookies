#ifndef PGM
#define PGM     const
#endif
#ifndef PGM_P
#define PGM_P   const char *
#endif
#ifndef bool
#define bool	_Bool
#endif

#include PLATFORM_HEADER
#include CONFIGURATION_HEADER

#include "hal/micro/generic/compiler/platform-common.h"

#include "stack/include/ember.h"

#include "em_cmu.h"
#include "imu.h"
#include "util.h"
#include "gps-uart.h"				// La gestion de la interrupcion de entrada por el uart del gps se realiza en com.c, en USART1_RX_IRQHandler(...)

#include "hal/hal.h"
#include "flex-bookkeeping.h"
#include "flex-callbacks.h"
#include "hal/plugin/serial/com.h"
#include "protocolored.h"
#include "wstk-sensors/wstk-sensors.h"
#include "em_rtcc.h"

#if defined(EMBER_AF_PLUGIN_MICRIUM_RTOS)
#include EMBER_AF_API_MICRIUM_RTOS
#endif

// Our entry point is typically main(), except in simulation.
// In simulation we don't include the cortexm3-specific headers.
#if defined(EMBER_TEST)
  #define MAIN nodeMain
  #if defined(EMBER_AF_API_DIAGNOSTIC_CORTEXM3)
    #undef EMBER_AF_API_DIAGNOSTIC_CORTEXM3
  #endif // EMBER_AF_API_DIAGNOSTIC_CORTEXM3
#else
  #define MAIN main
#endif // EMBER_TEST

// If serial functionality is enabled, we will initialize the serial ports
// during startup.  This has to happen after the HAL and gateway, if applicable,
// are initialized.
#ifdef EMBER_AF_API_SERIAL
  #include EMBER_AF_API_SERIAL

  #define SERIAL_INIT()                                                                                \
  do {                                                                                                 \
    emberSerialInit((uint8_t)APP_SERIAL, (SerialBaudRate)APP_BAUD_RATE, (SerialParity)PARITY_NONE, 1); \
  } while (false)
#else
  #define SERIAL_INIT()
  #define emberSerialPrintfLine(...)
#endif

// If printing is enabled, we will print some diagnostic information about the
// most recent reset and also during runtime.  On some platforms, extended
// diagnostic information is available.
#if defined(EMBER_AF_API_SERIAL) && defined(EMBER_AF_PRINT_ENABLE)
  #if defined(EMBER_AF_API_DIAGNOSTIC_CORTEXM3)
    #include EMBER_AF_API_DIAGNOSTIC_CORTEXM3
  #endif
static void printResetInformation(void);
  #define PRINT_RESET_INFORMATION printResetInformation
  #define emberAfGuaranteedPrint(...) \
  emberSerialGuaranteedPrintf(APP_SERIAL, __VA_ARGS__)
  #define emberAfGuaranteedPrintln(...)                   \
  do {                                                    \
    emberSerialGuaranteedPrintf(APP_SERIAL, __VA_ARGS__); \
    emberSerialGuaranteedPrintf(APP_SERIAL, "\r\n");      \
  } while (false)
#else
  #define PRINT_RESET_INFORMATION()
  #define emberAfGuaranteedPrint(...)
  #define emberAfGuaranteedPrintln(...)
#endif

EmberTaskId emAppTask;
extern const EmberEventData emAppEvents[];


//------------------------------------------------------------------------------
// Static functions.

#ifdef EMBER_AF_PRINT_ENABLE

static void printResetInformation(void)
{
  // Information about the most recent reset is printed during startup to aid
  // in debugging.
  emberAfGuaranteedPrintln("Reset info: 0x%x (%p)",
                           halGetResetInfo(),
                           halGetResetString());

//#if defined(EMBER_AF_API_DIAGNOSTIC_CORTEXM3) //Nov22
  emberAfGuaranteedPrintln("Extended reset info: 0x%2x (%p)",
                           halGetExtendedResetInfo(),
                           halGetExtendedResetString());
//#endif // EMBER_AF_API_DIAGNOSTIC_CORTEXM3	//Nov22
}

#endif // EMBER_AF_PRINT_ENABLE



/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: main.h
 *
 * MATLAB Coder version            : 5.1
 * C/C++ source code generated on  : 23-Nov-2022 18:02:41
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
#ifndef MAIN_H
#define MAIN_H

/* Include Files */
#include "flex-debug-print.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus

extern "C" {

#endif

  /* Function Declarations */
  extern int MAIN(MAIN_FUNCTION_PARAMETERS);

#ifdef __cplusplus

}
#endif
#endif

/*
 * File trailer for main.h
 *
 * [EOF]
 */
