/***************************************************************************//**
 * @file bspconfig.h
 * @brief Provide BSP (board support package) configuration parameters.
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

#ifndef BSPCONFIG_H
#define BSPCONFIG_H

#include "em_gpio.h"

#define BSP_STK
#define BSP_COOKIEBOARD

#define BSP_BCC_USART         USART0
#define BSP_BCC_CLK           cmuClock_USART0
#define BSP_BCC_TX_LOCATION   USART_ROUTELOC0_TXLOC_LOC0
#define BSP_BCC_RX_LOCATION   USART_ROUTELOC0_RXLOC_LOC0
#define BSP_BCC_TXPORT        gpioPortA
#define BSP_BCC_TXPIN         0
#define BSP_BCC_RXPORT        gpioPortA
#define BSP_BCC_RXPIN         1
#define BSP_BCC_ENABLE_PORT   gpioPortA
#define BSP_BCC_ENABLE_PIN    4                 /* VCOM_ENABLE */

#define BSP_IMU_RXPORT        gpioPortK
#define BSP_IMU_RXPIN         2
#define BSP_IMU_TXPORT        gpioPortK
#define BSP_IMU_TXPIN         0
#define BSP_IMU_MISOPORT      gpioPortK
#define BSP_IMU_MISOPIN       2
#define BSP_IMU_MOSIPORT      gpioPortK
#define BSP_IMU_MOSIPIN       0
#define BSP_IMU_SCLKPORT      gpioPortF
#define BSP_IMU_SCLKPIN       7
#define BSP_IMU_CSPORT        gpioPortK
#define BSP_IMU_CSPIN         1


#define BSP_DISP_ENABLE_PORT  gpioPortD
#define BSP_DISP_ENABLE_PIN   15                /* MemLCD display enable */

#define BSP_GPIO_LEDS
#define BSP_NO_OF_LEDS          2
#define BSP_LED_PORT    		gpioPortA
#define BSP_GPIO_LED0_PORT      gpioPortA
#define BSP_GPIO_LED0_PIN       5
#define BSP_GPIO_LED1_PORT      gpioPortA
#define BSP_GPIO_LED1_PIN       6
#define BSP_GPIO_LEDARRAY_INIT  { { BSP_GPIO_LED0_PORT, BSP_GPIO_LED0_PIN }, { BSP_GPIO_LED1_PORT, BSP_GPIO_LED1_PIN } }

#define BSP_INIT_DEFAULT  0

#if !defined(EMU_DCDCINIT_WSTK_DEFAULT)
/* Use emlib defaults */
#define EMU_DCDCINIT_WSTK_DEFAULT       EMU_DCDCINIT_DEFAULT
#endif

#if !defined(CMU_HFXOINIT_WSTK_DEFAULT)
#define CMU_HFXOINIT_WSTK_DEFAULT                                             \
  {                                                                           \
    false,      /* Low-noise mode for EFR32                                */ \
    false,      /* Disable auto-start on EM0/1 entry                       */ \
    false,      /* Disable auto-select on EM0/1 entry                      */ \
    false,      /* Disable auto-start and select on RAC wakeup             */ \
    _CMU_HFXOSTARTUPCTRL_CTUNE_DEFAULT,                                       \
    0x142,      /* Steady-state CTUNE for COOKIE boards without load caps */ \
    _CMU_HFXOSTEADYSTATECTRL_REGISH_DEFAULT,                                  \
    _CMU_HFXOSTARTUPCTRL_IBTRIMXOCORE_DEFAULT,                                \
    0x7,        /* Recommended steady-state XO core bias current           */ \
    0x6,        /* Recommended peak detection threshold                    */ \
    _CMU_HFXOTIMEOUTCTRL_SHUNTOPTTIMEOUT_DEFAULT,                             \
    0xA,        /* Recommended peak detection timeout                      */ \
    _CMU_HFXOTIMEOUTCTRL_STEADYTIMEOUT_DEFAULT,                               \
    _CMU_HFXOTIMEOUTCTRL_STARTUPTIMEOUT_DEFAULT,                              \
    cmuOscMode_Crystal,                                                       \
  }
#endif

#if !defined(RADIO_PTI_INIT)
#define RADIO_PTI_INIT                                                     \
  {                                                                        \
    RADIO_PTI_MODE_UART,    /* Simplest output mode is UART mode        */ \
    1600000,                /* Choose 1.6 MHz for best compatibility    */ \
    6,                      /* CookieBoard uses location 6 for DOUT         */ \
    gpioPortB,              /* Get the port for this loc                */ \
    12,                     /* Get the pin, location should match above */ \
    6,                      /* CookieBoard uses location 6 for DCLK         */ \
    gpioPortB,              /* Get the port for this loc                */ \
    11,                     /* Get the pin, location should match above */ \
    6,                      /* CookieBoard uses location 6 for DFRAME       */ \
    gpioPortB,              /* Get the port for this loc                */ \
    13,                     /* Get the pin, location should match above */ \
  }
#endif

#if !defined(RAIL_PTI_CONFIG)
#define RAIL_PTI_CONFIG                                                    \
  {                                                                        \
    RAIL_PTI_MODE_UART,     /* Simplest output mode is UART mode        */ \
    1600000,                /* Choose 1.6 MHz for best compatibility    */ \
    6,                      /* CookieBoard uses location 6 for DOUT         */ \
    gpioPortB,              /* Get the port for this loc                */ \
    12,                     /* Get the pin, location should match above */ \
    6,                      /* CookieBoard uses location 6 for DCLK         */ \
    gpioPortB,              /* Get the port for this loc                */ \
    11,                     /* Get the pin, location should match above */ \
    6,                      /* CookieBoard uses location 6 for DFRAME       */ \
    gpioPortB,              /* Get the port for this loc                */ \
    13,                     /* Get the pin, location should match above */ \
  }
#endif

#if !defined(RADIO_PA_2P4_INIT)
#define RADIO_PA_2P4_INIT                                    \
  {                                                          \
    PA_SEL_2P4_HP,    /* Power Amplifier mode             */ \
    PA_VOLTMODE_DCDC, /* Power Amplifier vPA Voltage mode */ \
    100,              /* Desired output power in dBm * 10 */ \
    0,                /* Output power offset in dBm * 10  */ \
    10,               /* Desired ramp time in us          */ \
  }
#endif

#if !defined(RAIL_PA_2P4_CONFIG)
#define RAIL_PA_2P4_CONFIG                                            \
  {                                                                   \
    RAIL_TX_POWER_MODE_2P4_HP, /* Power Amplifier mode             */ \
    1800,                      /* Power Amplifier vPA Voltage mode */ \
    10,                        /* Desired ramp time in us          */ \
  }
#endif

#if !defined(RADIO_PA_SUBGIG_INIT)
#define RADIO_PA_SUBGIG_INIT                                 \
  {                                                          \
    PA_SEL_SUBGIG,    /* Power Amplifier mode             */ \
    PA_VOLTMODE_DCDC, /* Power Amplifier vPA Voltage mode */ \
    100,              /* Desired output power in dBm * 10 */ \
    0,                /* Output power offset in dBm * 10  */ \
    10,               /* Desired ramp time in us          */ \
  }
#endif

#if !defined(RAIL_PA_SUBGIG_CONFIG)
#define RAIL_PA_SUBGIG_CONFIG                                         \
  {                                                                   \
    RAIL_TX_POWER_MODE_SUBGIG, /* Power Amplifier mode             */ \
    1800,                      /* Power Amplifier vPA Voltage mode */ \
    10,                        /* Desired ramp time in us          */ \
  }
#endif

#if !defined(RAIL_PA_DEFAULT_POWER)
#define RAIL_PA_DEFAULT_POWER 100
#endif

/***************************************************************************//**
 * @defgroup BOARD_Config_Settings BOARD module configuration
 * @{
 * @brief BOARD module configuration macro definitions
 ******************************************************************************/

#define BOARD_LED_PORT            gpioPortA       /**< LED port                         */
#define BOARD_LED_RED_PORT        gpioPortA       /**< Red LED port                     */
#define BOARD_LED_RED_PIN         5               /**< Red LED pin                      */
#define BOARD_LED_GREEN_PORT      gpioPortA       /**< Green LED port                   */
#define BOARD_LED_GREEN_PIN       6               /**< Green LED pin                    */

#define BOARD_PCOM_ENABLE_PORT       gpioPortF     /**< PCOM enable port                  */
#define BOARD_PCOM_ENABLE_PIN        8             /**< PCOM enable pin                   */

#define BOARD_TEMP_ENABLE_PORT       gpioPortF     /**< Temp sensor enable port    */
#define BOARD_TEMP_ENABLE_PIN        9             /**< Temp sensor enable pin     */


#define BOARD_IMU_ENABLE_PORT       gpioPortF     /**< IMU enable port                  */
#define BOARD_IMU_ENABLE_PIN        11             /**< IMU enable pin                   */
#define BOARD_IMU_INT1_PORT          gpioPortJ     /**< IMU interrupt port               */
#define BOARD_IMU_INT1_PIN           14            /**< IMU interrupt pin                */
#define BOARD_IMU_INT2_PORT          gpioPortJ     /**< IMU interrupt port               */
#define BOARD_IMU_INT2_PIN           15            /**< IMU interrupt pin                */
#define BOARD_IMU_SPI_PORT          gpioPortK     /**< IMU SPI port for CS, MISO & MOSI  */
#define BOARD_IMU_SPI_SCLK_PORT     gpioPortF     /**< IMU SPI SCLK port                 */
#define BOARD_IMU_SPI_MOSI_PIN      0             /**< IMU SPI master out slave in pin  */
#define BOARD_IMU_SPI_MISO_PIN      2             /**< IMU SPI master in slave out pin  */
#define BOARD_IMU_SPI_SCLK_PIN      7             /**< IMU SPI serial clock pin         */
#define BOARD_IMU_SPI_CS_PIN        1             /**< IMU SPI chip select pin          */


#define BOARD_CRYP_ENABLE_PORT       gpioPortF     /**< Crypto chip enable port           */
#define BOARD_CRYP_ENABLE_PIN        10            /**< Crypto chip enable pin            */

#define BOARD_USB_PGOOD_PORT    	 gpioPortC      /**< USB PGOOD port      */
#define BOARD_USB_PGOOD_PIN    		 6              /**< USB PGOOD pin      */

#define BOARD_IMU_I2C_ROUTELOC0		(I2C_ROUTELOC0_SDALOC_LOC15 | I2C_ROUTELOC0_SCLLOC_LOC15)
#define BOARD_TEMP_I2C_ROUTELOC0	(I2C_ROUTELOC0_SDALOC_LOC15 | I2C_ROUTELOC0_SCLLOC_LOC15)

#define BOARD_TEMP_I2C_PORT			gpioPortC
#define BOARD_TEMP_I2C_SDA_PIN		10
#define BOARD_TEMP_I2C_SCL_PIN		11

#define BOARD_APORT1_PORT           gpioPortC      /**< Analog port 1 port     */
#define BOARD_APORT1_X_CH0_PIN      0              /**< Analog port 1 X channel 0 pin      */
#define BOARD_APORT1_Y_CH1_PIN      1              /**< Analog port 1 Y channel 1 pin      */
#define BOARD_APORT1_X_CH2_PIN      2              /**< Analog port 1 X channel 2 pin      */
#define BOARD_APORT1_Y_CH3_PIN      3              /**< Analog port 1 Y channel 3 pin      */
#define BOARD_APORT1_X_CH4_PIN      4              /**< Analog port 1 X channel 4 pin      */
#define BOARD_APORT1_Y_CH5_PIN      5              /**< Analog port 1 Y channel 5 pin      */



/* External interrupts */
#define EXTI_INT2              15
#define EXTI_INT1              14

#define EXTI_IMU_INT1		EXTI_INT1
#define EXTI_IMU_INT2		EXTI_INT2


/** @} {end defgroup BOARD_Config_Setting} */

#define BSP_BCP_VERSION 2
#include "bsp_bcp.h"

#endif // BSPCONFIG_H
