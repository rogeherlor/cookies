# 13. Real IMU Bring-Up on Simplicity Studio v5


### Goal

This document records the first successful real-hardware IMU bring-up during the INARTRANS porting work.

Until this point, the portable application pipeline had been validated with fake GNSS and IMU samples. The next objective was to verify that the new Simplicity Studio v5 project could communicate with the real IMU mounted on the Cookie hardware.

The test confirms that the Cookie can initialise the IMU, read its device ID, and acquire real accelerometer and gyroscope samples.

---

## Current status

The real IMU bring-up is working.

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

When the board was moved, the IMU values changed as expected:

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

This confirms that the values are not fixed dummy data: they respond to real movement.

When the board is still, the accelerometer values are also physically coherent:

A ≈ -4, -20, 963 mg
G ≈ 0 dps

One accelerometer axis is close to ±1000 mg, which corresponds to gravity. The gyroscope values stay close to 0 dps, as expected when the board is not rotating.

Hardware interface

The old INARTRANS code uses an ICM20648 / ICM20948 inertial sensor.

The IMU is not accessed through I2C in this implementation. It is accessed through SPI, implemented using USART2 in synchronous mode.

The relevant configuration was found in the old file:

simplicity/Inartrans_v2_imu/icm20648_config.h

The SPI pin configuration is:

Signal	Port	Pin	Notes
MOSI	K	0	Master Out Slave In
MISO	K	2	Master In Slave Out
SCLK	F	7	SPI clock
CS	K	1	Chip select
IMU enable	F	11	Power/enable pin

The driver uses:

#define ICM20648_SPI_USART USART2
#define ICM20648_SPI_CLK   cmuClock_USART2

The old driver explicitly disables the I2C interface of the IMU and uses SPI:

ICM20648_registerWrite(ICM20648_REG_USER_CTRL, ICM20648_BIT_I2C_IF_DIS);

So, for this IMU bring-up, the relevant hardware path is:

ICM20648 / ICM20948
    ↓
SPI
    ↓
USART2 in synchronous mode
    ↓
Cookie application running on Simplicity Studio v5
Files copied from the old project

The following files were copied from the old Simplicity v4 INARTRANS IMU project into the new Simplicity v5 approach project:

porting/platform/imu_icm20648/icm20648_r.c
porting/platform/imu_icm20648/icm20648_r.h
porting/platform/imu_icm20648/icm20648_config.h

These files come from:

simplicity/Inartrans_v2_imu/

The purpose was to reuse the old, working ICM20648 driver while keeping it isolated from the rest of the portable application logic.

The driver is currently treated as legacy hardware code. The aim is not to rewrite it immediately, but to wrap it behind a cleaner platform adapter later.

Compatibility shims

The old driver expected some project-specific headers and helper functions from the old CookieBoard codebase.

Instead of copying the whole old board support layer, minimal compatibility shims were created.

A shim is a small compatibility layer. It lets old code find the names and functions it expects, while redirecting them to controlled implementations in the new project.

The goal is to avoid dragging unnecessary old dependencies into the new Simplicity v5 project.

cookieboard/util.h
#ifndef COOKIEBOARD_UTIL_H
#define COOKIEBOARD_UTIL_H

#include <stdint.h>

void UTIL_delay(uint32_t ms);

#endif
cookieboard/util.c
#include "cookieboard/util.h"

#include "sl_sleeptimer.h"

void UTIL_delay(uint32_t ms)
{
  sl_sleeptimer_delay_millisecond(ms);
}

Purpose:

Old driver call: UTIL_delay(ms)
New implementation: sl_sleeptimer_delay_millisecond(ms)

The old driver uses UTIL_delay(...) several times:

UTIL_delay(30)
UTIL_delay(100)
UTIL_delay(50)
UTIL_delay(5)

In the new project, this is mapped to Simplicity v5's sleeptimer delay function.

This avoids copying the full old util.c.

cookieboard/board.h
#ifndef COOKIEBOARD_BOARD_H
#define COOKIEBOARD_BOARD_H

#include <stdint.h>

#define BOARD_OK 0U

#endif

Purpose:

The old driver includes:

#include "cookieboard/board.h"

but for the current IMU bring-up only the BOARD_OK definition is required.

The full old board support file was not copied because it contains much more functionality than needed for this test.

cookieboard/icm20648.h
#ifndef COOKIEBOARD_ICM20648_H
#define COOKIEBOARD_ICM20648_H

#include "porting/platform/imu_icm20648/icm20648_r.h"

#endif

Purpose:

The old driver expects:

#include "cookieboard/icm20648.h"

The new project keeps the real driver header at:

porting/platform/imu_icm20648/icm20648_r.h

This shim redirects the old include path to the new driver location.

porting/platform/imu_icm20648/board_cookie.h
#ifndef BOARD_COOKIE_H
#define BOARD_COOKIE_H

#include <stdbool.h>
#include <stdint.h>

uint32_t BOARD_imuEnable(bool enable);

#endif
porting/platform/imu_icm20648/board_cookie.c
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

Purpose:

The old ICM20648 driver calls:

BOARD_imuEnable(false);
BOARD_imuEnable(true);

The original board_cookie.c contains much more functionality: LEDs, temperature sensor, I2C, GPIO interrupts, etc. For this bring-up, only IMU enable was required, so a minimal replacement was created.

This minimal implementation configures and controls:

IMU enable pin: PF11
Real IMU test wrapper

The test wrapper was added in the new Simplicity v5 project:

cookie_porting_real_imu.h
cookie_porting_real_imu.c

It performs the following sequence:

1. Print "Real IMU check started"
2. Call ICM20648_init()
3. Read the device ID using ICM20648_getDeviceID()
4. Enable accelerometer and gyroscope
5. Configure accelerometer full-scale range
6. Configure gyroscope full-scale range
7. Configure filter bandwidth
8. Read five accelerometer and gyroscope samples
9. Convert accelerometer values from g to mg
10. Print the samples

The successful device ID was:

0xE0

This indicates that the SPI communication is working and that the IMU is responding correctly.

The wrapper is intentionally simple. It is not the final IMU acquisition module.

Its only purpose is to prove that the real hardware can be accessed from the new Simplicity v5 project.

Current temporary entry point

The real IMU check is currently called from:

app_init.c

inside:

void ap_init(void)
{
  app_log_info("Inartrans porting app init\n");

  CookiePorting_RunRealImuCheck();
}

This is temporary.

The final application should not read only five IMU samples from ap_init(). This was only a hardware bring-up test.

At this stage, only one of the following checks should be active at a time:

CookiePorting_RunFakeAppFlow()
CookiePorting_RunRealImuCheck()

They should not be mixed without intention.

What this test proves

This test proves:

Simplicity Studio v5 project boots on the Cookie
Serial logging works
The old ICM20648 driver can be compiled in the new project
The IMU enable pin works
SPI using USART2 works
The IMU responds with a valid device ID
Real accelerometer and gyroscope values can be read
The values change when the board is moved

This is an important milestone because the project has moved from purely portable/fake-data tests to real hardware interaction.

What this test does not prove yet

This test does not yet prove:

Periodic IMU acquisition
200 Hz sampling
EKF prediction using real IMU data
GNSS acquisition from the real GPS module
GNSS + IMU fusion on hardware
Network transmission of the final packet
Reception by another Cookie
RSSI integration from real received/sent messages

Those are later integration steps.

Important interpretation of the IMU values

The stationary output:

A=-4,-20,968 mg
G=-1,1,0 dps

is coherent.

The accelerometer measures gravity even when the board is not moving. Therefore, one axis should usually be close to ±1000 mg, depending on the board orientation.

The gyroscope measures angular velocity. If the board is still, the values should be close to 0 dps.

When the board was moved, the output became:

A=-191,165,-665 mg
G=-173,189,-134 dps

This is also coherent because movement and rotation introduce significant accelerometer and gyroscope changes.

At this stage, the exact axis signs and orientation conventions are not the priority. The priority was to confirm that the values are alive and physically plausible.

Axis alignment will matter later, when connecting the real IMU to the EKF.

Next step

The next logical step is not to jump directly to radio or GNSS.

The next step should be:

Real IMU sample
    ↓
CookieImuSample
    ↓
CookieIMU_ConvertSample
    ↓
CookieIMU_PreprocessForNavigation
    ↓
CookieNavigation_PredictWithImu

This means connecting the real IMU values to the portable pipeline that has already been tested locally.

A first test can still run once from ap_init():

1. Initialise real IMU
2. Read one real sample
3. Convert it to CookieImuSample
4. Run the existing portable IMU conversion/preprocessing
5. Call CookieNavigation_PredictWithImu()
6. Print whether prediction succeeded

Only after that should periodic sampling be added.

Cleanup notes

The following items are temporary and should be cleaned later:

Item	Current role	Future decision
cookie_porting_fake.*	Fake full pipeline test	Keep as debug-only or remove once real pipeline works
cookie_porting_real_imu.*	Hardware bring-up test	Replace with a proper platform IMU adapter
cookieboard/* shims	Compatibility layer for old driver	Keep minimal or move under porting/platform/compat
Direct call from ap_init()	One-shot test	Replace with proper app/platform initialisation
Old sensornode logic	Communication example / inherited code	Avoid mixing until radio integration step

Recommended cleanup rule:

Do not delete working debug code immediately.
First isolate it, document it, and make sure only one debug entry point is active.
Proposed final direction

The final structure should separate:

portable application logic
hardware/platform adapters
legacy driver compatibility
Simplicity event/callback glue

Suggested direction:

porting/src/
    app/
    gnss/
    sensors/
    navigation/
    packets/
    network/

porting/platform/
    imu_icm20648/
        icm20648_r.c
        icm20648_r.h
        icm20648_config.h
        cookie_imu_platform.c
        cookie_imu_platform.h

    compat/
        util.c
        util.h
        board.h
        icm20648.h

The portable code should never depend directly on ICM20648_*.

Instead, the final application should use a platform adapter such as:

bool CookiePlatformImu_Init(void);

bool CookiePlatformImu_ReadSample(int32_t accel_mg[3],
                                  int32_t gyro_dps[3],
                                  uint32_t timestamp_ms);

That adapter can internally use the old ICM20648 driver.

The intended dependency direction should be:

portable app
    ↓
platform IMU adapter
    ↓
legacy ICM20648 driver
    ↓
Simplicity / EFR32 hardware

not:

portable app
    ↓
ICM20648 driver directly



# 12. Puesta en marcha de la IMU real en Simplicity Studio v5

## Objetivo

Este documento recoge la primera prueba correcta de la IMU real durante el trabajo de porting de INARTRANS.

Hasta este punto, el pipeline portable de la aplicación se había validado usando muestras falsas de GNSS e IMU. El siguiente objetivo era comprobar que el nuevo proyecto de Simplicity Studio v5 podía comunicarse con la IMU real montada en la Cookie.

La prueba confirma que la Cookie puede inicializar la IMU, leer su identificador de dispositivo y obtener muestras reales de acelerómetro y giroscopio.

Estado actual

La puesta en marcha de la IMU real funciona.

Salida observada por consola serie con la placa quieta:

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

Al mover la placa, los valores cambiaron como era esperable:

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

Esto confirma que los valores no son datos fijos ni simulados: responden al movimiento real de la placa.

Con la placa quieta, los valores también tienen sentido físico:

A ≈ -4, -20, 963 mg
G ≈ 0 dps

Una componente del acelerómetro está cerca de ±1000 mg, que corresponde a la gravedad. Los valores del giroscopio están cerca de 0 dps, como se espera cuando la placa no está rotando.

Interfaz hardware

El código antiguo de INARTRANS utiliza un sensor inercial ICM20648 / ICM20948.

En esta implementación, la IMU no se accede por I2C. Se accede mediante SPI, implementado usando USART2 en modo síncrono.

La configuración relevante se encontró en el archivo antiguo:

simplicity/Inartrans_v2_imu/icm20648_config.h

La configuración de pines SPI es:

Señal	Puerto	Pin	Descripción
MOSI	K	0	Master Out Slave In
MISO	K	2	Master In Slave Out
SCLK	F	7	Reloj SPI
CS	K	1	Chip select
IMU enable	F	11	Pin de alimentación/habilitación

El driver usa:

#define ICM20648_SPI_USART USART2
#define ICM20648_SPI_CLK   cmuClock_USART2

El driver antiguo deshabilita explícitamente la interfaz I2C de la IMU y usa SPI:

ICM20648_registerWrite(ICM20648_REG_USER_CTRL, ICM20648_BIT_I2C_IF_DIS);

Por tanto, para esta puesta en marcha de la IMU, el camino hardware relevante es:

ICM20648 / ICM20948
    ↓
SPI
    ↓
USART2 en modo síncrono
    ↓
Aplicación Cookie ejecutándose en Simplicity Studio v5
Archivos copiados desde el proyecto antiguo

Se copiaron los siguientes archivos desde el proyecto antiguo de IMU en Simplicity v4 al nuevo proyecto Simplicity v5:

porting/platform/imu_icm20648/icm20648_r.c
porting/platform/imu_icm20648/icm20648_r.h
porting/platform/imu_icm20648/icm20648_config.h

Proceden de:

simplicity/Inartrans_v2_imu/

El objetivo era reutilizar el driver antiguo de la IMU, que ya funcionaba, pero manteniéndolo aislado del resto de la lógica portable de la aplicación.

El driver se trata actualmente como código hardware heredado. La idea no es reescribirlo inmediatamente, sino envolverlo más adelante con un adaptador de plataforma más limpio.

Shims de compatibilidad

El driver antiguo esperaba algunas cabeceras y funciones auxiliares propias del antiguo código de CookieBoard.

En vez de copiar toda la capa antigua de soporte de placa, se crearon adaptadores mínimos.

Un “shim” es una capa pequeña de compatibilidad: permite que código antiguo encuentre los nombres y funciones que espera, pero redirigiéndolos a implementaciones nuevas y controladas.

La idea es evitar arrastrar dependencias antiguas innecesarias al nuevo proyecto Simplicity v5.

cookieboard/util.h
#ifndef COOKIEBOARD_UTIL_H
#define COOKIEBOARD_UTIL_H

#include <stdint.h>

void UTIL_delay(uint32_t ms);

#endif
cookieboard/util.c
#include "cookieboard/util.h"

#include "sl_sleeptimer.h"

void UTIL_delay(uint32_t ms)
{
  sl_sleeptimer_delay_millisecond(ms);
}

Objetivo:

Llamada antigua del driver: UTIL_delay(ms)
Implementación nueva: sl_sleeptimer_delay_millisecond(ms)

El driver antiguo usa UTIL_delay(...) varias veces:

UTIL_delay(30)
UTIL_delay(100)
UTIL_delay(50)
UTIL_delay(5)

En el proyecto nuevo, esto se traduce a la función de delay de sleeptimer de Simplicity v5.

Así evitamos copiar el util.c antiguo completo.

cookieboard/board.h
#ifndef COOKIEBOARD_BOARD_H
#define COOKIEBOARD_BOARD_H

#include <stdint.h>

#define BOARD_OK 0U

#endif

Objetivo:

El driver antiguo incluye:

#include "cookieboard/board.h"

pero para esta prueba mínima de IMU solo se necesitaba la definición BOARD_OK.

No se copió el soporte de placa antiguo completo porque contiene mucha más funcionalidad de la necesaria para esta prueba.

cookieboard/icm20648.h
#ifndef COOKIEBOARD_ICM20648_H
#define COOKIEBOARD_ICM20648_H

#include "porting/platform/imu_icm20648/icm20648_r.h"

#endif

Objetivo:

El driver antiguo espera:

#include "cookieboard/icm20648.h"

pero en el proyecto nuevo el header real del driver está en:

porting/platform/imu_icm20648/icm20648_r.h

Este shim redirige la ruta antigua al header nuevo.

porting/platform/imu_icm20648/board_cookie.h
#ifndef BOARD_COOKIE_H
#define BOARD_COOKIE_H

#include <stdbool.h>
#include <stdint.h>

uint32_t BOARD_imuEnable(bool enable);

#endif
porting/platform/imu_icm20648/board_cookie.c
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

Objetivo:

El driver antiguo llama a:

BOARD_imuEnable(false);
BOARD_imuEnable(true);

El board_cookie.c original contiene muchas más cosas: LEDs, sensor de temperatura, I2C, interrupciones GPIO, etc. Para esta prueba solo hacía falta habilitar la IMU, así que se creó una versión mínima.

Esta implementación mínima configura y controla:

Pin de enable de la IMU: PF11
Wrapper de prueba de IMU real

La prueba se añadió en el nuevo proyecto Simplicity v5:

cookie_porting_real_imu.h
cookie_porting_real_imu.c

Realiza la siguiente secuencia:

1. Imprime "Real IMU check started"
2. Llama a ICM20648_init()
3. Lee el device ID con ICM20648_getDeviceID()
4. Habilita acelerómetro y giroscopio
5. Configura el rango del acelerómetro
6. Configura el rango del giroscopio
7. Configura el ancho de banda del filtro
8. Lee cinco muestras de acelerómetro y giroscopio
9. Convierte aceleración de g a mg
10. Imprime las muestras

El device ID leído correctamente fue:

0xE0

Esto indica que la comunicación SPI funciona y que la IMU responde correctamente.

El wrapper es intencionadamente simple. No es el módulo final de adquisición de IMU.

Su único propósito es demostrar que se puede acceder al hardware real desde el nuevo proyecto Simplicity v5.

Punto de entrada temporal

La prueba de IMU real se llama actualmente desde:

app_init.c

dentro de:

void ap_init(void)
{
  app_log_info("Inartrans porting app init\n");

  CookiePorting_RunRealImuCheck();
}

Esto es temporal.

La aplicación final no debe limitarse a leer cinco muestras desde ap_init(). Esto solo era una prueba de puesta en marcha del hardware.

En este punto, solo debería haber una de estas llamadas activa cada vez:

CookiePorting_RunFakeAppFlow()
CookiePorting_RunRealImuCheck()

No deberían mezclarse sin intención.

Qué demuestra esta prueba

Esta prueba demuestra:

El proyecto Simplicity Studio v5 arranca en la Cookie
La salida por consola serie funciona
El driver antiguo ICM20648 puede compilarse en el proyecto nuevo
El pin de enable de la IMU funciona
El SPI mediante USART2 funciona
La IMU responde con un device ID válido
Se pueden leer valores reales de acelerómetro y giroscopio
Los valores cambian cuando se mueve la placa

Es un hito importante porque el proyecto pasa de pruebas puramente portables/fake-data a interacción con hardware real.

Qué no demuestra todavía

Esta prueba no demuestra todavía:

Adquisición periódica de IMU
Muestreo a 200 Hz
Predicción EKF usando datos reales de IMU
Adquisición GNSS desde el módulo GPS real
Fusión GNSS + IMU en hardware
Transmisión por radio del paquete final
Recepción por otra Cookie
Integración del RSSI real desde mensajes enviados/recibidos

Esos son pasos posteriores.

Interpretación importante de los valores IMU

La salida con la placa quieta:

A=-4,-20,968 mg
G=-1,1,0 dps

es coherente.

El acelerómetro mide la gravedad incluso cuando la placa no se mueve. Por eso, normalmente una de las componentes debería estar cerca de ±1000 mg, dependiendo de la orientación de la placa.

El giroscopio mide velocidad angular. Si la placa está quieta, sus valores deberían estar cerca de 0 dps.

Cuando la placa se movió, la salida pasó a valores como:

A=-191,165,-665 mg
G=-173,189,-134 dps

Esto también tiene sentido, porque el movimiento y la rotación introducen cambios significativos en acelerómetro y giroscopio.

En esta fase, los signos exactos de los ejes y la convención de orientación no son todavía la prioridad. La prioridad era confirmar que los valores están vivos y son físicamente plausibles.

La alineación de ejes será importante después, cuando se conecte la IMU real al EKF.

Siguiente paso

El siguiente paso lógico no es saltar directamente a radio o GNSS.

El siguiente paso debería ser:

Muestra real de IMU
    ↓
CookieImuSample
    ↓
CookieIMU_ConvertSample
    ↓
CookieIMU_PreprocessForNavigation
    ↓
CookieNavigation_PredictWithImu

Es decir: conectar los valores reales de la IMU al pipeline portable que ya estaba probado localmente.

Una primera prueba puede seguir ejecutándose una vez desde ap_init():

1. Inicializar la IMU real
2. Leer una muestra real
3. Convertirla a CookieImuSample
4. Ejecutar la conversión/preprocesado IMU portable
5. Llamar a CookieNavigation_PredictWithImu()
6. Imprimir si la predicción ha funcionado

Solo después tendría sentido añadir muestreo periódico.

Notas de limpieza

Los siguientes elementos son temporales y deberían limpiarse más adelante:

Elemento	Función actual	Decisión futura
cookie_porting_fake.*	Prueba fake del pipeline completo	Mantener como debug-only o eliminar cuando el pipeline real funcione
cookie_porting_real_imu.*	Prueba de hardware bring-up	Sustituir por un adaptador real de plataforma
cookieboard/* shims	Compatibilidad para el driver antiguo	Mantener mínimos o mover a porting/platform/compat
Llamada directa desde ap_init()	Test puntual	Sustituir por inicialización real de plataforma/app
Lógica antigua de sensornode	Ejemplo heredado de comunicación	No mezclar hasta la fase de radio

Regla de limpieza recomendada:

No borrar código de debug que funciona inmediatamente.
Primero aislarlo, documentarlo y asegurarse de que solo hay un punto de entrada de debug activo.
Dirección final propuesta

La estructura final debería separar:

lógica portable de aplicación
adaptadores hardware/plataforma
compatibilidad con drivers antiguos
glue/callbacks de Simplicity

Dirección sugerida:

porting/src/
    app/
    gnss/
    sensors/
    navigation/
    packets/
    network/

porting/platform/
    imu_icm20648/
        icm20648_r.c
        icm20648_r.h
        icm20648_config.h
        cookie_imu_platform.c
        cookie_imu_platform.h

    compat/
        util.c
        util.h
        board.h
        icm20648.h

El código portable no debería depender directamente de ICM20648_*.

En su lugar, la aplicación final debería usar un adaptador de plataforma como:

bool CookiePlatformImu_Init(void);

bool CookiePlatformImu_ReadSample(int32_t accel_mg[3],
                                  int32_t gyro_dps[3],
                                  uint32_t timestamp_ms);

Ese adaptador puede usar internamente el driver antiguo ICM20648.

La dirección de dependencia debería ser:

aplicación portable
    ↓
adaptador de plataforma IMU
    ↓
driver legado ICM20648
    ↓
Simplicity / hardware EFR32

y no:

aplicación portable
    ↓
driver ICM20648 directamente