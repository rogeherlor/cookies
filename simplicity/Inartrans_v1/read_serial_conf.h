/*
 * read_serial_conf.h
 *
 *  Created on: 7 de jul. de 2022
 *      Author: PCNET22Win10 Rogelio
 */

//#include "serial.h"

#ifndef READ_SERIAL_CONF_H_
#define READ_SERIAL_CONF_H_
#endif /* READ_SERIAL_CONF_H_ */

#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>

#include PLATFORM_HEADER
#include "stack/include/ember-types.h"
#include "stack/include/error.h"

//Host processors do not use Ember Message Buffers.
#ifndef EZSP_HOST
  #include "stack/include/packet-buffer.h"
#endif

#include "hal/hal.h"
#include "serial.h"

#include <stdarg.h>

#ifdef EMBER_SERIAL_USE_STDIO
#include <stdio.h>
#endif //EMBER_SERIAL_USE_STDIO

#ifdef EMBER_SERIAL_CUSTOM_STDIO
#include EMBER_SERIAL_CUSTOM_STDIO
#define EMBER_SERIAL_USE_STDIO
#endif // EMBER_SERIAL_CUSTOM_STDIO

// AppBuilder and Afv2 will define the characteristics of the Serial ports here.
#if defined(ZA_GENERATED_HEADER)
  #include ZA_GENERATED_HEADER
#endif

#include "stack/include/ember.h"
#include "em_device.h"
#include "em_chip.h"
#include "bsp.h"
//#include "command-interpreter/command-interpreter2.h"
#include "debug-print/debug-print.h"


void read_serial_conf(void);
