#include "cookieboard/util.h"

#include "sl_sleeptimer.h"

void UTIL_delay(uint32_t ms)
{
  sl_sleeptimer_delay_millisecond(ms);
}
