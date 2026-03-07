/***************************************************************************//**
 * @file board_cookie.h
 * @brief BOARD module header file
 * @version 5.6.0
 *******************************************************************************
 * # License
 * <b>Copyright 2017 Silicon Laboratories, Inc. http://www.silabs.com</b>
 *******************************************************************************
 *
 * This file is licensed under the Silicon Labs License Agreement. See the file
 * "Silabs_License_Agreement.txt" for details. Before using this software for
 * any purpose, you must agree to the terms of that agreement.
 *
 ******************************************************************************/

#ifndef BOARD_COOKIE_H
#define BOARD_COOKIE_H

#include <stdint.h>

typedef void (*BOARD_IrqCallback)(void);/**< Interrupt callback function type definition */

/**************************************************************************//**
* @addtogroup CookieBoard_BSP
* @{
******************************************************************************/

/***************************************************************************//**
 * @addtogroup BOARD_COOKIE
 * @{
 ******************************************************************************/

uint32_t BOARD_init(void);
 
uint32_t BOARD_i2cBusSelect(uint8_t select);

void BOARD_ledSet(uint8_t leds);

uint32_t BOARD_imuEnable(bool enable);
uint32_t BOARD_imuEnableIRQ1(bool enable);
uint32_t BOARD_imuEnableIRQ2(bool enable);
void BOARD_imuSetIRQ1Callback(BOARD_IrqCallback cb);
void BOARD_imuSetIRQ2Callback(BOARD_IrqCallback cb);
void BOARD_imuClearIRQ1(void);
void BOARD_imuClearIRQ2(void);

static void gpioInterruptHandler(uint8_t pin);

uint32_t BOARD_TempEnable(bool enable);


/** @} */
/** @} */

#endif // BOARD_COOKIE_H
