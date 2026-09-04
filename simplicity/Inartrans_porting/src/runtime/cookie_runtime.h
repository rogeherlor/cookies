#ifndef COOKIE_RUNTIME_H
#define COOKIE_RUNTIME_H

#include <stdbool.h>

/*
 * Runtime glue for the embedded Simplicity application.
 *
 * This module connects the hardware-specific platform adapters
 * with the portable CookieApp logic.
 *
 */

/*
 * Initialise the runtime and all currently enabled hardware inputs.
 *
 * Returns true if the runtime was initialised correctly.
 */
bool CookieRuntime_Init(void);

/*
 * Periodic runtime processing.
 *
 * This function must be called frequently from the application tick.
 * It decides internally whether enough time has passed to read a new
 * IMU sample.
 */
void CookieRuntime_Process(void);

#endif /* COOKIE_RUNTIME_H */
