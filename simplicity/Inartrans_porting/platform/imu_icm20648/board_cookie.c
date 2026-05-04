#include "board_cookie.h"

#include "em_cmu.h"
#include "em_gpio.h"

#define BOARD_OK 0U

#define BOARD_IMU_ENABLE_PORT gpioPortF
#define BOARD_IMU_ENABLE_PIN  11

uint32_t BOARD_imuEnable(bool enable)
{
  CMU_ClockEnable(cmuClock_GPIO, true);

  GPIO_PinModeSet(BOARD_IMU_ENABLE_PORT,
                  BOARD_IMU_ENABLE_PIN,
                  gpioModePushPull,
                  0);

  if (enable) {
    GPIO_PinOutSet(BOARD_IMU_ENABLE_PORT, BOARD_IMU_ENABLE_PIN);
  } else {
    GPIO_PinOutClear(BOARD_IMU_ENABLE_PORT, BOARD_IMU_ENABLE_PIN);
  }

  return BOARD_OK;
}
