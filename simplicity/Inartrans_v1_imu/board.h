/***************************************************************************//**
 * @file board.h
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

#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include "gpiointerrupt.h"

#include "bspconfig.h"

/***************************************************************************//**
 * @defgroup CookieBoard_BSP Cookieboard BSP
 * @{
 * @brief BSP for CookieBoard
 ******************************************************************************/

/**************************************************************************//**
* @name BOARD Error Codes
* @{
******************************************************************************/
#define BOARD_OK                              0     /**< OK                                        */
#define BOARD_ERROR_I2C_TRANSFER_TIMEOUT      0x01  /**< I2C timeout occurred                      */
#define BOARD_ERROR_I2C_TRANSFER_NACK         0x02  /**< No acknowledgement received               */
#define BOARD_ERROR_I2C_TRANSFER_FAILED       0x03  /**< I2C transaction failed                    */
#define BOARD_ERROR_PIC_ID_MISMATCH           0x04  /**< The ID of the PIC is invalid              */
#define BOARD_ERROR_PIC_FW_INVALID            0x05  /**< Invalid PIC firmware                      */
#define BOARD_ERROR_PIC_FW_UPDATE_FAILED      0x06  /**< PIC firmware update failed                */

#define BOARD_ERROR_NO_POWER_INT_CTRL         0x10  /**< Power and Interrupt Controller not found  */
#define BOARD_ERROR_I2C_BUS_SELECT_INVALID    0x11  /**< Invalid I2C bus selection                 */
#define BOARD_ERROR_I2C_BUS_SELECT_FAILED     0x12  /**< I2C bus selection failed                  */
/**@}*/
/**@}*/

//cambiar lo del I2C
/** @cond DO_NOT_INCLUDE_WITH_DOXYGEN */
#define BOARD_I2C_BUS_SELECT_NONE          0           /**< No I2C bus selected                               	*/
#define BOARD_I2C_BUS_SELECT_TEMP_SENSOR		  (1 << 0)    /**< The I2C bus of the temperature sensor selected 		*/
#define BOARD_I2C_BUS_SELECT_IMU          (1 << 1)    /**< The I2C bus of the 6axis inertial sensor selected    */
/** @endcond */

uint32_t BOARD_init                (void);

//uint32_t BOARD_imuEnable           (bool enable);
//uint32_t BOARD_imuEnableIRQ        (bool enable);
//void     BOARD_imuClearIRQ         (void);

//void     BOARD_flashDeepPowerDown  (void);

//uint32_t BOARD_bapEnable           (bool enable);

uint32_t BOARD_TempEnable        (bool enable);

void     BOARD_ledSet              (uint8_t leds);

//uint32_t BOARD_micEnable           (bool enable);

uint32_t BOARD_i2cBusSelect        (uint8_t select);

//uint8_t  BOARD_pushButtonGetState  (void);
//void     BOARD_pushButtonEnableIRQ (bool enable);

#ifdef BSP_COOKIE
  #include "board_cookie.h"
#endif

#endif // BOARD_H
