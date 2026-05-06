# 13. Real IMU Bring-Up on Simplicity Studio v5

## 13.1 Purpose

This document records the first successful real-hardware IMU bring-up during the INARTRANS porting work.

Until this point, the portable application pipeline had been validated using fake GNSS and IMU samples. The next objective was to verify that the new Simplicity Studio v5 project could communicate with the real IMU mounted on the Cookie hardware.

This bring-up confirms that the Cookie can:

- initialise the real IMU;
- read its device ID;
- acquire real accelerometer and gyroscope samples;
- detect physical movement through changing IMU values;
- run periodic IMU acquisition from the embedded runtime.

---

## 13.2 Current status

The real IMU bring-up is working.

The project now has two validated stages:

1. A one-shot hardware check, used to prove that the IMU can be initialised and read.
2. A periodic runtime check, used to prove that the IMU can be sampled continuously at the target acquisition rate.

The current runtime reads the IMU every 5 ms, equivalent to 200 Hz.

To keep the serial output readable, the runtime prints one debug line every 200 samples.

```txt
IMU sampling rate: 200 Hz
Debug print rate:  approximately 1 Hz
```

---

## 13.3 Hardware interface

The old INARTRANS code uses an ICM20648 / ICM20948 inertial sensor.

In this implementation, the IMU is not accessed through I2C. It is accessed through SPI, implemented using USART2 in synchronous mode.

The relevant configuration was found in the old file:

```txt
simplicity/Inartrans_v2_imu/icm20648_config.h
```

### 13.3.1 SPI pin configuration

| Signal | Port | Pin | Description |
|---|---:|---:|---|
| MOSI | K | 0 | Master Out Slave In |
| MISO | K | 2 | Master In Slave Out |
| SCLK | F | 7 | SPI clock |
| CS | K | 1 | Chip select |
| IMU enable | F | 11 | IMU power / enable pin |

The driver uses:

```c
#define ICM20648_SPI_USART USART2
#define ICM20648_SPI_CLK   cmuClock_USART2
```

The old driver explicitly disables the I2C interface of the IMU and uses SPI:

```c
ICM20648_registerWrite(ICM20648_REG_USER_CTRL, ICM20648_BIT_I2C_IF_DIS);
```

Therefore, the hardware path for this bring-up is:

```txt
ICM20648 / ICM20948
        ↓
SPI
        ↓
USART2 in synchronous mode
        ↓
Cookie application running on Simplicity Studio v5
```

---

## 13.4 Driver files reused from the old project

The following files were copied from the old Simplicity v4 INARTRANS IMU project into the new Simplicity v5 porting project:

```txt
porting/platform/imu_icm20648/icm20648_r.c
porting/platform/imu_icm20648/icm20648_r.h
porting/platform/imu_icm20648/icm20648_config.h
```

They come from:

```txt
simplicity/Inartrans_v2_imu/
```

The purpose was to reuse the old working ICM20648 driver while keeping it isolated from the portable application logic.

The driver is currently treated as legacy hardware code. The aim is not to rewrite it immediately, but to wrap it behind a cleaner platform adapter.

---

## 13.5 Compatibility shims

The old driver expected some project-specific headers and helper functions from the old CookieBoard codebase.

Instead of copying the whole old board support layer, minimal compatibility shims were created.

A shim is a small compatibility layer. It lets old code find the names and functions it expects, while redirecting them to controlled implementations in the new project.

The goal is to avoid dragging unnecessary old dependencies into the new Simplicity v5 project.

---

### 13.5.1 `cookieboard/util.h` and `cookieboard/util.c`

The old driver uses:

```c
UTIL_delay(ms);
```

A minimal replacement was created.

#### 13.5.1.1 `cookieboard/util.h`

```c
#ifndef COOKIEBOARD_UTIL_H
#define COOKIEBOARD_UTIL_H

#include <stdint.h>

void UTIL_delay(uint32_t ms);

#endif
```

#### 13.5.1.2 `cookieboard/util.c`

```c
#include "cookieboard/util.h"

#include "sl_sleeptimer.h"

void UTIL_delay(uint32_t ms)
{
  sl_sleeptimer_delay_millisecond(ms);
}
```

Purpose:

```txt
Old driver call:     UTIL_delay(ms)
New implementation: sl_sleeptimer_delay_millisecond(ms)
```

The old driver uses `UTIL_delay(...)` several times, for example:

```c
UTIL_delay(30);
UTIL_delay(100);
UTIL_delay(50);
UTIL_delay(5);
```

In the new project, these calls are mapped to the Simplicity v5 sleeptimer delay function.

This avoids copying the full old `util.c`.

---

### 13.5.2 `cookieboard/board.h`

The old driver includes:

```c
#include "cookieboard/board.h"
```

For the current IMU bring-up, only `BOARD_OK` was required, so a minimal shim was created.

#### 13.5.2.1 `cookieboard/board.h`

```c
#ifndef COOKIEBOARD_BOARD_H
#define COOKIEBOARD_BOARD_H

#include <stdint.h>

#define BOARD_OK 0U

#endif
```

The full old board support file was not copied because it contains much more functionality than needed for this step.

---

### 13.5.3 `cookieboard/icm20648.h`

The old driver expects:

```c
#include "cookieboard/icm20648.h"
```

The new project keeps the actual driver header at:

```txt
porting/platform/imu_icm20648/icm20648_r.h
```

So this shim redirects the old include path to the new driver location.

#### 13.5.3.1 `cookieboard/icm20648.h`

```c
#ifndef COOKIEBOARD_ICM20648_H
#define COOKIEBOARD_ICM20648_H

#include "porting/platform/imu_icm20648/icm20648_r.h"

#endif
```

---

### 13.5.4 `porting/platform/imu_icm20648/board_cookie.h` and `board_cookie.c`

The old ICM20648 driver calls:

```c
BOARD_imuEnable(false);
BOARD_imuEnable(true);
```

The original `board_cookie.c` contains much more functionality: LEDs, temperature sensor, I2C, GPIO interrupts, etc.

For this bring-up, only the IMU enable pin was required, so a minimal replacement was created.

#### 13.5.4.1 `porting/platform/imu_icm20648/board_cookie.h`

```c
#ifndef BOARD_COOKIE_H
#define BOARD_COOKIE_H

#include <stdbool.h>
#include <stdint.h>

uint32_t BOARD_imuEnable(bool enable);

#endif
```

#### 13.5.4.2 `porting/platform/imu_icm20648/board_cookie.c`

```c
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
```

This minimal implementation configures and controls:

```txt
IMU enable pin: PF11
```

---

## 13.6 One-shot real IMU test

The first real-hardware test was implemented as a temporary wrapper:

```txt
cookie_porting_real_imu.h
cookie_porting_real_imu.c
```

It performs the following sequence:

1. Print `Real IMU check started`.
2. Call `ICM20648_init()`.
3. Read the device ID using `ICM20648_getDeviceID()`.
4. Enable accelerometer and gyroscope.
5. Configure accelerometer full-scale range.
6. Configure gyroscope full-scale range.
7. Configure filter bandwidth.
8. Read five accelerometer and gyroscope samples.
9. Convert accelerometer values from `g` to `mg`.
10. Print the samples.

The successful device ID was:

```txt
0xE0
```

This indicates that SPI communication is working and that the IMU is responding correctly.

This wrapper is intentionally simple. It is not the final IMU acquisition module. Its purpose was only to prove that the real hardware can be accessed from the new Simplicity v5 project.

---

## 13.7 One-shot test output

### 13.7.1 Board still

Observed serial output with the board still:

```txt
Direct Mode Device
Network up
Inartrans porting app init
Real IMU check started
ICM20648 init OK
IMU device ID: 0xE0
IMU sample 1: A=-4,-20,968 mg | G=-1,1,0 dps
IMU sample 2: A=-4,-20,962 mg | G=0,0,0 dps
IMU sample 3: A=-2,-20,961 mg | G=-1,0,0 dps
IMU sample 4: A=-3,-19,963 mg | G=-1,1,0 dps
IMU sample 5: A=-3,-21,963 mg | G=-1,1,0 dps
Real IMU check finished
```

The values are physically coherent:

```txt
A ≈ -4, -20, 963 mg
G ≈ 0 dps
```

One accelerometer axis is close to ±1000 mg, which corresponds to gravity. The gyroscope values stay close to 0 dps, as expected when the board is not rotating.

---

### 13.7.2 Board moved

When the board was moved, the IMU values changed as expected:

```txt
Direct Mode Device
Network up
Inartrans porting app init
Real IMU check started
ICM20648 init OK
IMU device ID: 0xE0
IMU sample 1: A=-8,86,-584 mg | G=-135,147,-121 dps
IMU sample 2: A=-79,76,-636 mg | G=-134,151,-108 dps
IMU sample 3: A=-191,165,-665 mg | G=-173,189,-134 dps
IMU sample 4: A=-100,282,-715 mg | G=-207,193,-164 dps
IMU sample 5: A=95,363,-770 mg | G=-209,209,-176 dps
Real IMU check finished
```

This confirms that the values are not fixed dummy data. They respond to real movement.

At this stage, the exact axis signs and orientation conventions are not the priority. The priority was to confirm that the values are alive and physically plausible.

Axis alignment will matter later, when connecting the real IMU data to the EKF.

---

## 13.8 Clean platform IMU adapter

After the one-shot test, the direct use of the ICM20648 driver was wrapped behind a cleaner platform-facing adapter:

```txt
porting/platform/imu_icm20648/cookie_imu_platform.h
porting/platform/imu_icm20648/cookie_imu_platform.c
```

The public interface is:

```c
typedef struct {
  int32_t accel_mg[3];
  int32_t gyro_dps[3];
  uint32_t timestamp_ms;
  bool valid;
} CookiePlatformImuSample;

bool CookiePlatformImu_Init(void);

bool CookiePlatformImu_ReadSample(CookiePlatformImuSample *sample);

uint8_t CookiePlatformImu_GetDeviceId(void);
```

The important design decision is that the rest of the application should not call `ICM20648_*` directly.

Instead, the dependency direction should be:

```txt
portable application
        ↓
platform IMU adapter
        ↓
legacy ICM20648 driver
        ↓
Simplicity / EFR32 hardware
```

and not:

```txt
portable application
        ↓
ICM20648 driver directly
```

This keeps the old driver isolated and makes the portable code easier to test.

---

## 13.9 Embedded runtime integration

Once the platform adapter was working, the IMU was integrated into a runtime module:

```txt
porting/src/runtime/cookie_runtime.h
porting/src/runtime/cookie_runtime.c
```

The goal of this module is to keep Simplicity callbacks small.

Instead of putting IMU, GNSS, packet and navigation logic directly inside `app_init.c` or `app_process.c`, the Simplicity glue should only call:

```c
CookieRuntime_Init();
CookieRuntime_Process();
```

The runtime currently performs:

- IMU initialisation during `CookieRuntime_Init()`;
- periodic IMU acquisition from `CookieRuntime_Process()`;
- forwarding of each IMU sample into the portable `CookieApp` layer;
- debug printing at a reduced rate.

---

## 13.10 Current temporary entry points

The runtime is currently started from:

```txt
app_init.c
```

inside:

```c
void app_init(void)
{
  app_log_info("Inartrans porting app init\n");

  CookieRuntime_Init();
}
```

The runtime process function is called from:

```txt
app_process.c
```

inside:

```c
void emberAfTickCallback(void)
{
  CookieRuntime_Process();
}
```

This replaces the earlier one-shot call:

```c
CookiePorting_RunRealImuCheck();
```

That one-shot function is still useful as a bring-up/debug example, but it should not be the normal runtime path.

At this stage, only one IMU entry point should be active:

| Entry point | Role |
|---|---|
| `CookiePorting_RunRealImuCheck()` | One-shot hardware debug |
| `CookieRuntime_Init()` / `CookieRuntime_Process()` | Periodic runtime acquisition |

They should not be mixed unless there is a specific reason.

---

## 13.11 Periodic IMU acquisition

The runtime currently reads the IMU every 5 ms:

```c
#define COOKIE_RUNTIME_IMU_PERIOD_MS 5u
```

This corresponds to:

```txt
1000 ms / 5 ms = 200 samples per second
```

So the current runtime acquisition rate is:

```txt
200 Hz
```

To avoid flooding the serial console, only one debug line is printed every 200 samples:

```c
#define COOKIE_RUNTIME_IMU_DEBUG_EVERY_N_SAMPLES 200u
```

This means the IMU is sampled at 200 Hz, but only logged at approximately 1 Hz.

---

## 13.12 Periodic runtime output

Observed serial output:

```txt
Direct Mode Device
Network up
Inartrans porting app init
CookieRuntime: init started
CookiePlatformImu: init started
CookiePlatformImu: device ID = 0xE0
CookiePlatformImu: init OK
CookieRuntime: IMU ready, device ID = 0xE0
CookieRuntime: IMU period = 5 ms, debug every 200 samples
CookieRuntime: init OK
```

Periodic IMU output:

```txt
CookieRuntime: IMU #200 t=1174 ms | A=14,-7,970 mg | G=-1,0,0 dps
CookieRuntime: IMU #400 t=2174 ms | A=12,-8,969 mg | G=-1,0,0 dps
CookieRuntime: IMU #600 t=3174 ms | A=13,-10,970 mg | G=-1,0,0 dps
CookieRuntime: IMU #800 t=4174 ms | A=15,-7,972 mg | G=-1,0,0 dps
CookieRuntime: IMU #1000 t=5174 ms | A=14,-6,972 mg | G=-1,0,0 dps
CookieRuntime: IMU #1200 t=6174 ms | A=15,-7,971 mg | G=-1,0,0 dps
CookieRuntime: IMU #1400 t=7174 ms | A=15,-7,970 mg | G=-1,0,0 dps
CookieRuntime: IMU #1600 t=8174 ms | A=13,-9,972 mg | G=-1,0,0 dps
CookieRuntime: IMU #1800 t=9174 ms | A=14,-7,972 mg | G=-1,0,0 dps
CookieRuntime: IMU #2000 t=10174 ms | A=17,-7,970 mg | G=-1,0,0 dps
CookieRuntime: IMU #2200 t=11174 ms | A=13,-8,970 mg | G=-1,0,0 dps
CookieRuntime: IMU #2400 t=12174 ms | A=14,-7,971 mg | G=-1,0,0 dps
CookieRuntime: IMU #2600 t=13174 ms | A=13,-6,972 mg | G=-1,0,0 dps
```

This confirms the intended periodic behaviour.

Every 200 samples, the timestamp increases by approximately 1000 ms:

```txt
#200  -> t = 1174 ms
#400  -> t = 2174 ms
#600  -> t = 3174 ms
#800  -> t = 4174 ms
```

Therefore:

```txt
200 samples    ≈ 1000 ms
1 sample       ≈ 5 ms
sampling rate  ≈ 200 Hz
```

This is the target IMU acquisition rate used in the old INARTRANS code.

---

## 13.13 Timing implementation

The runtime uses the Simplicity sleeptimer to obtain a millisecond timestamp:

```c
static uint32_t runtime_now_ms(void)
{
  uint32_t ticks = sl_sleeptimer_get_tick_count();
  return sl_sleeptimer_tick_to_ms(ticks);
}
```

This helper hides Silicon Labs timing details from the rest of the runtime logic.

The IMU period check uses unsigned subtraction:

```c
if ((uint32_t)(now_ms - last_imu_read_ms) >= COOKIE_RUNTIME_IMU_PERIOD_MS) {
  last_imu_read_ms = now_ms;
  runtime_process_imu(now_ms);
}
```

This is intentional. Unsigned subtraction makes the comparison robust across timer wrap-around.

---

## 13.14 Connection to the portable pipeline

Each real IMU sample is pushed into the portable application layer through:

```c
CookieApp_ProcessImuSample(&runtime_app,
                           platform_sample.accel_mg,
                           platform_sample.gyro_dps,
                           platform_sample.timestamp_ms);
```

The intended IMU path is now:

```txt
Real ICM20648 sample
        ↓
CookiePlatformImuSample
        ↓
CookieApp_ProcessImuSample()
        ↓
CookieImuSample
        ↓
CookieIMU_ConvertSample()
        ↓
CookieIMU_PreprocessForNavigation()
        ↓
CookieNavigation_PredictWithImu()
```

At this stage, the real IMU is already entering the portable application layer.

The navigation prediction will only be meaningful once the navigation module has been initialised with a valid GNSS fix or equivalent initial state.

Therefore, periodic IMU acquisition is working, but full GNSS + IMU navigation fusion is not complete yet.

---

## 13.15 What this milestone proves

This milestone proves that:

- the Simplicity Studio v5 project boots on the Cookie;
- serial logging works;
- the old ICM20648 driver can be compiled in the new project;
- the IMU enable pin works;
- SPI using USART2 works;
- the IMU responds with a valid device ID;
- real accelerometer and gyroscope values can be read;
- the values change when the board is moved;
- a clean platform adapter can hide the legacy driver;
- the runtime can acquire IMU samples periodically;
- the current 5 ms period produces approximately 200 Hz sampling;
- the real IMU samples can be forwarded into the portable `CookieApp` pipeline.

This is an important milestone because the project has moved from fake-data tests to real hardware interaction while preserving a clean separation between portable logic and platform-specific code.

---

## 13.16 What this milestone does not prove yet

This milestone does not yet prove:

- GNSS acquisition from the real GPS module;
- GNSS epoch detection in the new runtime;
- GPS-synchronised timestamps;
- EKF prediction with a fully initialised navigation state;
- GNSS + IMU fusion on hardware;
- construction of final real packets from live GNSS + IMU data;
- radio transmission of the final packet;
- reception by another Cookie;
- integration of real RSSI into the packet logic.

Those are later integration steps.

---

## 13.17 Important interpretation of IMU values

A stationary output such as:

```txt
A=14,-7,970 mg
G=-1,0,0 dps
```

is coherent.

The accelerometer measures gravity even when the board is not moving. Therefore, one component should usually be close to ±1000 mg, depending on the board orientation.

The gyroscope measures angular velocity. If the board is still, its values should stay close to 0 dps.

When the board is moved, values such as:

```txt
A=-191,165,-665 mg
G=-173,189,-134 dps
```

are also coherent because movement and rotation introduce significant accelerometer and gyroscope changes.

At this stage, the exact axis signs and orientation conventions are not the priority. The priority was to confirm that the values are alive, periodic and physically plausible.

Axis alignment will matter later when the IMU data is connected to the real EKF.

---

## 13.18 Files involved

### 13.18.1 Platform IMU files

```txt
porting/platform/imu_icm20648/board_cookie.c
porting/platform/imu_icm20648/board_cookie.h
porting/platform/imu_icm20648/cookie_imu_platform.c
porting/platform/imu_icm20648/cookie_imu_platform.h
porting/platform/imu_icm20648/icm20648_config.h
porting/platform/imu_icm20648/icm20648_r.c
porting/platform/imu_icm20648/icm20648_r.h
```

Role:

```txt
Real IMU hardware access and legacy ICM20648 driver isolation.
```

---

### 13.18.2 Compatibility shim files

```txt
cookieboard/util.c
cookieboard/util.h
cookieboard/board.h
cookieboard/icm20648.h
```

Role:

```txt
Minimal compatibility layer required by the old driver.
```

These files should remain small. If they start growing too much, that is a sign that old board-support logic is being dragged into the new project.

---

### 13.18.3 Runtime files

```txt
porting/src/runtime/cookie_runtime.c
porting/src/runtime/cookie_runtime.h
```

Role:

```txt
Embedded runtime glue between Simplicity callbacks, real hardware adapters and portable CookieApp logic.
```

---

### 13.18.4 Portable application files touched by the IMU path

```txt
porting/src/app/app.c
porting/src/app/app.h
porting/src/sensors/imu_sample.c
porting/src/sensors/imu_sample.h
porting/src/sensors/imu_converter.c
porting/src/sensors/imu_converter.h
porting/src/sensors/imu_preprocessor.c
porting/src/sensors/imu_preprocessor.h
porting/src/navigation/navigation.c
porting/src/navigation/navigation.h
```

Role:

```txt
Portable processing of IMU samples after they have been read from hardware.
```

These files should not depend directly on the ICM20648 driver.

---

### 13.18.5 Temporary debug/example files

```txt
cookie_porting_real_imu.c
cookie_porting_real_imu.h
cookie_porting_fake.c
cookie_porting_fake.h
```

Role:

```txt
Temporary debug helpers and bring-up checks.
```

They are useful while porting, but they should not become the final application architecture.

---

## 13.19 Cleanup notes

The following items are temporary or should remain isolated:

| Item | Current role | Future decision |
|---|---|---|
| `cookie_porting_fake.*` | Fake full-pipeline test | Keep as debug-only or move to examples |
| `cookie_porting_real_imu.*` | One-shot hardware bring-up test | Keep as example or remove once runtime is stable |
| `cookieboard/*` shims | Compatibility layer for old driver | Keep minimal or move under `porting/platform/compat` |
| Direct hardware calls from app callbacks | Early bring-up style | Avoid in final code |
| Old sensornode / AI example logic | Inherited communication/example code | Keep disabled or outside the active build |
| Direct `ICM20648_*` use outside platform layer | Legacy coupling | Avoid |

Recommended cleanup rule:

```txt
Do not delete working debug code immediately.
First isolate it, document it, and make sure only one debug entry point is active.
```

---

## 13.20 Current repository structure

The repository does not contain the full Simplicity Studio workspace. 
Only the files needed to document and reproduce the porting approach are versioned.

Current relevant structure:

```txt
simplicity/Inartrans_porting/
    docs/
    examples/
        simplicity_real_imu_check/

    cookieboard/
        board.h
        icm20648.h
        util.c
        util.h

    platform/
        imu_icm20648/
            board_cookie.c
            board_cookie.h
            cookie_imu_platform.c
            cookie_imu_platform.h
            icm20648_config.h
            icm20648_r.c
            icm20648_r.h

    src/
        app/
        gnss/
        sensors/
        navigation/
        packets/
        network/
        runtime/
```
The cookieboard/ directory contains minimal compatibility shims required by the old ICM20648 driver. It does not represent the full old CookieBoard support layer.

## 13.21 Periodic IMU acquisition at 200 Hz

After the one-shot bring-up test, the IMU was connected to the embedded runtime.

The runtime now reads one IMU sample every 5 ms, equivalent to 200 Hz. To keep the serial output readable, only one sample every 200 readings is printed.

Observed output:

```txt
CookieRuntime: IMU #200 t=1174 ms | raw A=12,8,968 mg | raw G=-1,0,0 dps | dt=0.005 s | nav_ready=0 | predicted=0
CookieRuntime: converted A=0.12,0.08,9.50 m/s2 | G=-0.017,0.000,0.000 rad/s
CookieRuntime: navigation input A=-0.12,-9.50,-0.08 m/s2 | G=0.017,0.000,0.000 rad/s
```

Current runtime flow:

```txt
real IMU hardware
        ↓
platform adapter
        ↓
portable IMU sample
        ↓
conversion
        ↓
preprocessing
        ↓
navigation input
```

At this stage, navigation prediction is not expected to run yet because the EKF is only initialised after receiving enough valid GNSS fixes. Therefore, the observed values:

```txt
nav_ready=0
predicted=0
```

are correct for the current integration step.
