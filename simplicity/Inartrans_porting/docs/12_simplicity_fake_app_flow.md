# Simplicity Fake App Flow

## Purpose

This document records the first successful execution of the portable `CookieApp` flow inside a real Simplicity Studio v5 project running on the Cookie hardware.

The goal of this step was not to use real sensors or real radio transmission yet.

The goal was to verify that the portable architecture can compile, flash and run inside the embedded Simplicity environment.

## Context

The clean portable code lives in:

    ~/cookies/simplicity/Inartrans_porting/

The Simplicity bring-up project used for this test lives in:

    ~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/

This Simplicity project was copied from a Silicon Labs Connect Direct Mode Device example.

It is currently used as an integration laboratory, not as the final cleaned project.

## What was tested

The test executed the portable app flow inside the Cookie using fake inputs:

    Fake NMEA GNSS epoch
            ↓
    CookieApp_ProcessGnssEpoch()

    Fake IMU samples
            ↓
    CookieApp_ProcessImuSample()

    CookieApp state
            ↓
    CookieApp_BuildDataMessage()
            ↓
    message[90]

The generated message contains:

    network header[15] + data packet[75]

## Modules included

The following portable modules were copied into the Simplicity approach project under:

    porting/src/

Included modules:

- `src/gnss`
- `src/sensors`
- `src/navigation`
- `src/packets`
- `src/network`
- `src/app`

For this bring-up test, the navigation module still used the mock EKF:

    porting/src/navigation/mock_ekf.c

The real EKF was not integrated yet.

## Important build issue: duplicate `main`

After copying the portable app folder, the Simplicity project tried to compile:

    porting/src/app/main.c

This caused a linker error because the Simplicity project already has its own `main.c`.

The error was:

    multiple definition of `main`

The reason is that Simplicity controls the embedded startup through its own main function:

    sl_system_init();
    ap_init();
    sl_system_process_action();

Therefore, the portable `main.c` must not be compiled inside Simplicity.

For this test, it was disabled by renaming it to:

    porting/src/app/main_pc_only.c.disabled

The portable app is called from the Simplicity application layer instead.

## Simplicity entry point used

The fake app flow was called from:

    app_init.c

Specifically from:

    ap_init()

The purpose was to execute the portable app once during startup and print confirmation through `app_log_info()`.

## Serial/debug output

The firmware was built, flashed to the Cookie and observed through the serial console using:

    screen /dev/ttyACM0 115200

The successful output was:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake app flow started
    GNSS fake epochs processed
    IMU fake samples processed
    CookieApp fake message built successfully
    Message size: 90 bytes
    Packet type: 3
    Payload validity byte: A
    Payload GNSS mode byte: 7
    Payload end markers: ,,
    >

Some unreadable characters may appear at reset before the normal log starts. This is not considered a problem at this stage because the actual log is readable afterwards.

## What this validates

This test validates that:

- A Simplicity Studio v5 project can be built for the Cookie.
- The generated `.hex` can be flashed to the device.
- Serial/debug output is available.
- Custom code can be called from `ap_init()`.
- The portable modules can compile inside the embedded build environment.
- The portable `CookieApp` flow can run on the Cookie hardware.
- Fake GNSS data can be processed.
- Fake IMU data can be processed.
- The navigation wrapper can run with the mock EKF.
- The 75-byte legacy-compatible data packet can be built.
- The 90-byte network frame can be built.
- The generated message has the expected key fields.

## What this does NOT validate

This test does not validate:

- Real GNSS UART input
- Real IMU hardware readings
- Real timestamps from Simplicity
- Real EKF integration
- Physical navigation correctness
- Radio transmission
- Radio reception
- Discovery/request/repair/config network logic
- Runtime node ID / PAN ID / rank management
- Power behaviour
- Long-term execution stability

## Why this step matters

This step confirms that the architecture developed and tested locally is not only valid on PC, but can also run inside the actual embedded/Simplicity environment.

This is an important intermediate milestone before connecting real hardware inputs.

The system has moved from:

    PC tests only

to:

    portable code running on the Cookie through Simplicity Studio

## Current status after this step

Completed:

- Simplicity v5 project builds.
- Firmware flashes to the Cookie.
- Logs are visible through serial.
- Basic custom fake check works.
- Full portable fake app flow works.
- `message[90]` is generated on the Cookie.

Still pending:

- Replace mock EKF with real EKF.
- Connect real IMU input.
- Connect real GNSS UART input.
- Send `message[90]` through the radio stack.
- Reintroduce network-management logic step by step.

## Recommended next step

The next technical step should be chosen carefully.

Two reasonable options are:

1. Integrate the real EKF.
2. Connect one real hardware input, preferably the IMU first.

The IMU may be a better first hardware input than GNSS because GNSS requires UART buffering and NMEA epoch detection, which adds more moving parts.

____________________________________________________
____________________________________________________

# Flujo fake de app en Simplicity

## Objetivo

Este documento registra la primera ejecución correcta del flujo portable de `CookieApp` dentro de un proyecto real de Simplicity Studio v5 ejecutándose en el hardware de la Cookie.

El objetivo de este paso no era usar sensores reales ni transmisión radio real todavía.

El objetivo era comprobar que la arquitectura portable puede compilarse, flashearse y ejecutarse dentro del entorno embedded de Simplicity.

## Contexto

El código portable limpio está en:

    ~/cookies/simplicity/Inartrans_porting/

El proyecto de bring-up en Simplicity usado para esta prueba está en:

    ~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/

Este proyecto de Simplicity se copió a partir de un ejemplo de Silicon Labs Connect Direct Mode Device.

Actualmente se usa como laboratorio de integración, no como proyecto final limpio.

## Qué se ha probado

La prueba ejecutó el flujo de la app portable dentro de la Cookie usando entradas falsas:

    Época GNSS NMEA falsa
            ↓
    CookieApp_ProcessGnssEpoch()

    Muestras IMU falsas
            ↓
    CookieApp_ProcessImuSample()

    Estado de CookieApp
            ↓
    CookieApp_BuildDataMessage()
            ↓
    message[90]

El mensaje generado contiene:

    cabecera de red[15] + paquete de datos[75]

## Módulos incluidos

Los siguientes módulos portables se copiaron dentro del proyecto approach de Simplicity bajo:

    porting/src/

Módulos incluidos:

- `src/gnss`
- `src/sensors`
- `src/navigation`
- `src/packets`
- `src/network`
- `src/app`

Para esta prueba de bring-up, el módulo de navegación todavía usaba el EKF mock:

    porting/src/navigation/mock_ekf.c

El EKF real aún no se ha integrado.

## Problema importante de build: `main` duplicado

Después de copiar la carpeta portable de app, el proyecto de Simplicity intentó compilar:

    porting/src/app/main.c

Esto produjo un error de linker porque el proyecto de Simplicity ya tiene su propio `main.c`.

El error fue:

    multiple definition of `main`

La razón es que Simplicity controla el arranque embedded mediante su propia función main:

    sl_system_init();
    ap_init();
    sl_system_process_action();

Por tanto, el `main.c` portable no debe compilarse dentro de Simplicity.

Para esta prueba, se desactivó renombrándolo a:

    porting/src/app/main_pc_only.c.disabled

La app portable se llama desde la capa de aplicación de Simplicity.

## Punto de entrada usado en Simplicity

El flujo fake de app se llamó desde:

    app_init.c

Concretamente desde:

    ap_init()

El objetivo era ejecutar una vez la app portable durante el arranque e imprimir confirmación mediante `app_log_info()`.

## Salida serial/debug

El firmware se compiló, se flasheó en la Cookie y se observó mediante consola serial usando:

    screen /dev/ttyACM0 115200

La salida correcta fue:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake app flow started
    GNSS fake epochs processed
    IMU fake samples processed
    CookieApp fake message built successfully
    Message size: 90 bytes
    Packet type: 3
    Payload validity byte: A
    Payload GNSS mode byte: 7
    Payload end markers: ,,
    >

Pueden aparecer caracteres ilegibles al reset antes de que empiece el log normal. En esta fase no se considera un problema porque después la salida legible aparece correctamente.

## Qué valida este test

Este test valida que:

- Un proyecto de Simplicity Studio v5 puede compilar para la Cookie.
- El `.hex` generado puede flashearse en el dispositivo.
- La salida serial/debug está disponible.
- Se puede llamar a código propio desde `ap_init()`.
- Los módulos portables pueden compilar dentro del entorno embedded.
- El flujo portable de `CookieApp` puede ejecutarse en el hardware de la Cookie.
- Los datos GNSS fake pueden procesarse.
- Los datos IMU fake pueden procesarse.
- El navigation wrapper puede ejecutarse con el EKF mock.
- El paquete de datos legacy-compatible de 75 bytes puede construirse.
- La trama de red de 90 bytes puede construirse.
- El mensaje generado contiene los campos clave esperados.

## Qué NO valida este test

Este test no valida:

- Entrada UART real del GNSS
- Lecturas reales del hardware IMU
- Timestamps reales desde Simplicity
- Integración del EKF real
- Corrección física de la navegación
- Transmisión radio
- Recepción radio
- Lógica network discovery/request/repair/config
- Gestión runtime de node ID / PAN ID / rango
- Comportamiento energético
- Estabilidad de ejecución a largo plazo

## Por qué importa este paso

Este paso confirma que la arquitectura desarrollada y probada localmente no solo es válida en PC, sino que también puede ejecutarse dentro del entorno real embedded/Simplicity.

Es un hito intermedio importante antes de conectar entradas hardware reales.

El sistema ha pasado de:

    tests solo en PC

a:

    código portable ejecutándose en la Cookie mediante Simplicity Studio

## Estado actual después de este paso

Completado:

- El proyecto de Simplicity v5 compila.
- El firmware se flashea en la Cookie.
- Los logs son visibles por serial.
- La comprobación fake mínima funciona.
- El flujo fake completo de app portable funciona.
- `message[90]` se genera en la Cookie.

Pendiente:

- Sustituir el EKF mock por el EKF real.
- Conectar entrada IMU real.
- Conectar entrada UART GNSS real.
- Enviar `message[90]` mediante el stack de radio.
- Reintroducir la lógica de gestión de red paso a paso.

## Próximo paso recomendado

El siguiente paso técnico debe elegirse con cuidado.

Dos opciones razonables son:

1. Integrar el EKF real.
2. Conectar una entrada hardware real, preferiblemente primero la IMU.

La IMU puede ser una mejor primera entrada hardware que el GNSS porque GNSS requiere buffering UART y detección de época NMEA, lo que añade más piezas móviles.