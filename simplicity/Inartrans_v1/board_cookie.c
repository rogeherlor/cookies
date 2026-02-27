/***************************************************************************//**
 * @file board_cookie.c
 * @brief BOARD module source file
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

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "i2cspm.h"
#include "em_cmu.h"
#include "em_gpio.h"
#include "em_prs.h"
#include "em_timer.h"
#include "em_usart.h"
#include "bspconfig.h"

#include "board.h"
#include "util.h"

#include "board_cookie.h"

/**************************************************************************//**
* @addtogroup CookieBoard_BSP
* @{
******************************************************************************/

/***************************************************************************//**
 * @defgroup BOARD_COOKIE BOARD Module for CookieBoard
 * @{
 * @brief Board hardware control, configuraton and miscellaneous functions
 * @details This module contains functions releated to board features. It allows
 *  control over power management features, interrupt controller and sensors
 *
 * The BOARD module uses the common I2CSPM driver to communicate with the
 * I2C sensors on the board. The following board features can be enabled
 * when needed using the BOARD Module:
 * - RH/Temp (Si7021)
 * - Inertial sensor (ICM-20948)
 *
 *
 ******************************************************************************/

/** @cond DO_NOT_INCLUDE_WITH_DOXYGEN */

#define BOARD_ENABLED_NONE      0
#define BOARD_ENABLED_TEMP    	(1 << 0)
#define BOARD_ENABLED_IMU       (1 << 1)


/*******************************************************************************
 *******************************   TYPEDEFS   **********************************
 ******************************************************************************/

typedef struct {
  uint16_t enabled;
  uint16_t busInUse;
} BOARD_SensorInfo_t;

/****************************************************************************/
/* Local variables                                                          */
/****************************************************************************/

static BOARD_SensorInfo_t sensorInfo;
static BOARD_IrqCallback imuIRQCallback;

/****************************************************************************/
/* Local function prototypes                                                */
/****************************************************************************/
static void gpioInterruptHandler(uint8_t pin);

/** @endcond */

/***************************************************************************//**
 * @brief
 *    Initializes the Cookieboard
 *
 * @return
 *    Returns zero on OK, non-zero otherwise
 ******************************************************************************/
uint32_t BOARD_init(void)
{

  I2CSPM_Init_TypeDef  i2cInit        = I2CSPM_INIT_DEFAULT;
  uint32_t status;

  i2cInit.port            = I2C0;
  i2cInit.sclPort         = gpioPortC;
  i2cInit.sclPin          = 11;
  i2cInit.sdaPort         = gpioPortC;
  i2cInit.sdaPin          = 10;
  i2cInit.portLocationScl = 15;
  i2cInit.portLocationSda = 15;
  I2CSPM_Init(&i2cInit);

  sensorInfo.enabled = 0x00;
  sensorInfo.busInUse = 0x00;

  status = BOARD_OK;

  /* Enable GPIO clock */
  CMU_ClockEnable(cmuClock_GPIO, true);

  /* Initialize LEDs */
  GPIO_PinModeSet(BOARD_LED_RED_PORT, BOARD_LED_RED_PIN, gpioModePushPull, 0);
  GPIO_PinModeSet(BOARD_LED_GREEN_PORT, BOARD_LED_GREEN_PIN, gpioModePushPull, 0);

  /* Configure the Interrupt pins */
  GPIO_PinModeSet(BOARD_IMU_INT1_PORT, BOARD_IMU_INT1_PIN, gpioModeInput, 0);
  GPIO_PinModeSet(BOARD_IMU_INT2_PORT, BOARD_IMU_INT2_PIN, gpioModeInput, 0);

  /*********************************************************************/
  /** IMU pin config                                                  **/
  /*********************************************************************/
  GPIO_PinModeSet(BOARD_IMU_ENABLE_PORT, BOARD_IMU_ENABLE_PIN, gpioModePushPull, 0);
  GPIO_PinModeSet(BOARD_IMU_SPI_PORT, BOARD_IMU_SPI_MOSI_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(BOARD_IMU_SPI_PORT, BOARD_IMU_SPI_MISO_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(BOARD_IMU_SPI_PORT, BOARD_IMU_SPI_SCLK_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(BOARD_IMU_SPI_PORT, BOARD_IMU_SPI_CS_PIN, gpioModeDisabled, 0);

  /*********************************************************************/
  /** ENV Sensor pin config                                           **/
  /*********************************************************************/
  GPIO_PinModeSet(BOARD_TEMP_ENABLE_PORT, BOARD_TEMP_ENABLE_PIN, gpioModePushPull, 0);
  GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SDA_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SCL_PIN, gpioModeDisabled, 0);


  GPIOINT_Init();

  return status;
}

/***************************************************************************//**
 * @brief
 *    Turns on or off the red and/or green LED
 *
 * @param[in] leds
 *    The two LSB bits determine the state of the green and red LED. If the bit
 *    is 1 then the LED will be turned on.
 *
 * @return
 *    None
 ******************************************************************************/
void BOARD_ledSet(uint8_t leds)
{
  uint8_t pins[2] = { BOARD_LED_GREEN_PIN, BOARD_LED_RED_PIN };
  for ( int i = 0; i < 2; i++ ) {
    if ( ( (leds >> i) & 1) == 1 ) {
      GPIO_PinOutSet(BOARD_LED_PORT, pins[i]);
    } else {
      GPIO_PinOutClear(BOARD_LED_PORT, pins[i]);
    }
  }
  return;
}

/***************************************************************************//**
 * @brief
 *    Enables or disables the accelerometer and gyroscope sensor
 *
 * @param[in] enable
 *    Set true to enable, false to disable
 *
 * @return
 *    Returns zero on OK, non-zero otherwise
 ******************************************************************************/
uint32_t BOARD_imuEnable(bool enable)
{
  if ( enable ) {
    GPIO_PinOutSet(BOARD_IMU_ENABLE_PORT, BOARD_IMU_ENABLE_PIN);
    sensorInfo.enabled |= BOARD_ENABLED_IMU;
  } else {
    GPIO_PinOutClear(BOARD_IMU_ENABLE_PORT, BOARD_IMU_ENABLE_PIN);
    sensorInfo.enabled &= ~BOARD_ENABLED_IMU;
  }

  return BOARD_OK;
}

/***************************************************************************//**
 * @brief
 *    Enables or disables the accelerometer and gyroscope GPIO interrupts
 *
 * @param[in] enable
 *    Set true to enable, false to disable
 *
 * @return
 *    Returns zero on OK, non-zero otherwise
 ******************************************************************************/
uint32_t BOARD_imuEnableIRQ1(bool enable)
{
  GPIO_ExtIntConfig(BOARD_IMU_INT1_PORT, BOARD_IMU_INT1_PIN, EXTI_IMU_INT1, false, true, enable);

  return BOARD_OK;
}

uint32_t BOARD_imuEnableIRQ2(bool enable)
{
  GPIO_ExtIntConfig(BOARD_IMU_INT2_PORT, BOARD_IMU_INT2_PIN, EXTI_IMU_INT2, false, true, enable);

  return BOARD_OK;
}
/***************************************************************************//**
 * @brief
 *    Enables or disables the temperature sensor
 *
 * @param[in] enable
 *    Set true to enable, false to disable
 *
 * @return
 *    Returns zero on OK, non-zero otherwise
 ******************************************************************************/
uint32_t BOARD_TempEnable(bool enable)
{
  if ( enable ) {
    /* Enable power */
    GPIO_PinOutSet(BOARD_TEMP_ENABLE_PORT, BOARD_TEMP_ENABLE_PIN);
    /* Setup I2C pins */
    GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SDA_PIN, gpioModeWiredAnd, 1);
    GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SCL_PIN, gpioModeWiredAnd, 1);
    sensorInfo.enabled |= BOARD_ENABLED_TEMP;
  } else {
    /* Disable power */
    GPIO_PinOutClear(BOARD_TEMP_ENABLE_PORT, BOARD_TEMP_ENABLE_PIN);
    /* Disconnect I2C pins */
    GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SDA_PIN, gpioModeDisabled, 0);
    GPIO_PinModeSet(BOARD_TEMP_I2C_PORT, BOARD_TEMP_I2C_SCL_PIN, gpioModeDisabled, 0); 
	sensorInfo.enabled &= ~BOARD_ENABLED_TEMP;
  }
  
  return BOARD_OK;
}
/***************************************************************************//**
 * @brief
 *    Sets up the route register of the I2C device to use the correct
 *    set of pins
 *
 * * @param[in] select
 *    The I2C bus route to use (None, Environmental sensors, 6-axis inertial sensor
 *
 * @return
 *    Returns zero on OK, non-zero otherwise
 ******************************************************************************/
uint32_t BOARD_i2cBusSelect(uint8_t select)
{
  uint32_t status;

  status = BOARD_OK;

  switch ( select ) {
    case BOARD_I2C_BUS_SELECT_NONE:
      I2C0->ROUTEPEN  = 0;
      I2C0->ROUTELOC0 = 0;
      sensorInfo.busInUse = BOARD_I2C_BUS_SELECT_NONE;
      break;

    case BOARD_I2C_BUS_SELECT_TEMP_SENSOR:
      I2C0->ROUTELOC0 = BOARD_TEMP_I2C_ROUTELOC0;
      I2C0->ROUTEPEN = (I2C_ROUTEPEN_SCLPEN | I2C_ROUTEPEN_SDAPEN);
      sensorInfo.busInUse = BOARD_I2C_BUS_SELECT_TEMP_SENSOR;
      break;

    case BOARD_I2C_BUS_SELECT_IMU:
      I2C0->ROUTELOC0 = BOARD_IMU_I2C_ROUTELOC0;
      I2C0->ROUTEPEN = (I2C_ROUTEPEN_SCLPEN | I2C_ROUTEPEN_SDAPEN);
      sensorInfo.busInUse = BOARD_I2C_BUS_SELECT_IMU;
      break;

    default:
      status = BOARD_ERROR_I2C_BUS_SELECT_INVALID;
  }

  return status;
}


/***************************************************************************//**
 * @brief
 *    Functions to register the IMU sensor interrupt callback functions
 *
 * @param[in] cb
 *    The callback function to be registered
 *
 * @return
 *    Returns none
 *****************************************************************************/
void BOARD_imuSetIRQ1Callback(BOARD_IrqCallback cb)
{
  GPIOINT_CallbackRegister(EXTI_IMU_INT1, gpioInterruptHandler);
  imuIRQCallback = cb;

  return;
}

void BOARD_imuSetIRQ2Callback(BOARD_IrqCallback cb)
{
  GPIOINT_CallbackRegister(EXTI_IMU_INT2, gpioInterruptHandler);
  imuIRQCallback = cb;

  return;
}

/***************************************************************************//**
 * @brief
 *    Functions to clear the IMU sensor interrupts
 *
 * @return
 *    Returns none
 ******************************************************************************/
void BOARD_imuClearIRQ1(void)
{
  GPIO_IntClear(1 << EXTI_IMU_INT1);

  return;
}
void BOARD_imuClearIRQ2(void)
{
  GPIO_IntClear(1 << EXTI_IMU_INT2);

  return;
}



/** @cond DO_NOT_INCLUDE_WITH_DOXYGEN */



/***************************************************************************//**
 * @brief
 *    Common callback from the GPIOINT driver
 *
 * @return
 *    Returns none
 ******************************************************************************/
static void gpioInterruptHandler(uint8_t pin)
{
  BOARD_IrqCallback callback;

  switch ( pin ) {
       break;
    case EXTI_IMU_INT1:
		callback = imuIRQCallback;
		break;
	case EXTI_IMU_INT2:
		callback = imuIRQCallback;
		break;
     default:
      callback = NULL;
  }

  if (callback != NULL) {
    callback();
  }

  return;
}
/** @endcond */

/** @} (end defgroup BOARD_COOKIE) */
/** @} {end addtogroup CookieBoard_BSP} */
