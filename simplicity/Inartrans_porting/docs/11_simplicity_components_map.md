# Simplicity Components Map

## Purpose

This document explains the role of Simplicity Studio in the Inartrans_porting project and maps the different hardware/software communication layers involved.

The goal is to avoid confusing several different types of communication:

- GNSS communication
- IMU communication
- Radio communication between Cookies
- Debug/log communication with the PC

These are different interfaces and they solve different problems.

## Current situation

The portable code has already been developed and tested locally in the repository:

    cookies/simplicity/Inartrans_porting/

This portable code includes:

- GNSS parser and converter
- IMU sample, converter and preprocessor
- Navigation wrapper
- Packet builder
- Network frame builder
- App orchestration

This code runs in local PC tests using fake inputs and a mock EKF.

The next step is to integrate it into a real Simplicity Studio project so that it can run on the Cookie hardware.

## Two different worlds

There are currently two separate working areas.

### 1. Portable code repository

Located at:

    ~/cookies/simplicity/Inartrans_porting/

This contains the clean, modular and testable code.

It should remain as hardware-independent as possible.

### 2. Simplicity Studio workspace project

Located at:

    ~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/

This is a real Simplicity Studio v5 project copied from a Silicon Labs Connect Direct Mode Device example.

It compiles, flashes and runs on the Cookie hardware.

It is currently used as a bring-up laboratory, not as the final cleaned project.

## What has already been verified

The Simplicity project has been successfully tested with a minimal fake check.

Verified:

- The project builds with 0 errors.
- The generated `.hex` file can be flashed to the Cookie.
- Serial output can be seen using `screen`.
- Custom code can be called from `ap_init()`.
- `app_log_info()` output is visible.
- The old experimental loop was disabled to avoid continuous failed transmissions.

Observed serial output:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake check executed
    >

This confirms that the current Simplicity project is a valid starting point for integration experiments.

## Important idea: not all communication is the same

The project uses several communication interfaces, but they are not interchangeable.

Each one connects different parts of the system.

    GNSS module  → Cookie MCU       via UART
    IMU sensor   → Cookie MCU       via I2C/SPI or driver
    Cookie MCU   → other Cookie     via radio / Connect / Ember
    Cookie MCU   → PC terminal      via UART / virtual COM / debug console

So the question is not:

    Should the project use UART, I2C, SPI or radio?

The correct question is:

    Which interface is used for each boundary of the system?

## Communication interfaces

## UART

### What UART is

UART stands for Universal Asynchronous Receiver/Transmitter.

It is a simple serial communication interface.

It usually uses two main lines:

- TX: transmit
- RX: receive

UART sends bytes sequentially, one after another.

It is commonly used for:

- GPS/GNSS modules
- Debug serial terminals
- Communication between a microcontroller and a PC
- Simple peripheral modules

### Why UART matters here

The GNSS module outputs NMEA text sentences.

Example:

    $GNRMC,123519,A,4807.038,N,01131.000,E,...

This is text, sent byte by byte.

The Cookie receives this GNSS text through a UART interface.

The old project used UART code in `gps-uart.c`.

The new portable code does not directly read UART. Instead, it expects a complete NMEA epoch:

    CookieApp_ProcessGnssEpoch(buffer, length)

Therefore, the Simplicity-specific adapter must:

1. Receive characters from the GNSS UART.
2. Store them in a buffer.
3. Detect when a complete GNSS epoch has arrived.
4. Pass that text buffer to the portable app.

### UART in this project

Used for:

    GNSS module → Cookie

Possibly also used for:

    Cookie → PC debug console

Important:

UART is not the radio link between Cookies.

## I2C

### What I2C is

I2C stands for Inter-Integrated Circuit.

It is a communication bus commonly used to connect sensors to a microcontroller.

It typically uses two lines:

- SDA: data
- SCL: clock

Multiple devices can share the same I2C bus, each with a different address.

### Why I2C matters here

Many IMU sensors communicate through I2C.

The Cookie board may use an IMU driver that internally uses I2C to read acceleration and gyroscope data.

The portable code should not care whether the IMU is internally connected through I2C or SPI.

It only expects logical IMU values:

    accel_mg[3]
    gyro_dps[3]
    timestamp_ms

### I2C in this project

Potentially used for:

    IMU sensor → Cookie

Important:

I2C is not used for communication between Cookies.

It is normally used inside the device, between the MCU and sensors.

## SPI

### What SPI is

SPI stands for Serial Peripheral Interface.

It is another communication interface commonly used between a microcontroller and peripherals.

It usually uses:

- MOSI: master out, slave in
- MISO: master in, slave out
- SCLK: clock
- CS: chip select

SPI is often faster than I2C, but usually needs more wires.

### Why SPI matters here

Some IMU sensors can communicate using SPI instead of I2C.

Whether the Cookie IMU uses I2C or SPI depends on the board and driver configuration.

For the portable code, this detail should remain hidden.

The Simplicity adapter or IMU driver should provide samples to the app in the expected format.

### SPI in this project

Potentially used for:

    IMU sensor → Cookie

Important:

The portable IMU modules do not need to know if the physical sensor uses SPI or I2C.

## Radio / Connect / Ember

### What the radio stack is

The Cookie communicates wirelessly with other Cookies or a coordinator using the Silicon Labs radio stack.

The current Simplicity example is based on:

    Connect: Direct Mode Device

This provides APIs such as:

    emberNetworkInit()
    emberMessageSend(...)
    emberAfIncomingMessageCallback(...)
    emberAfMessageSentCallback(...)
    emberAfStackStatusCallback(...)

### What Direct Mode means

Direct Mode means that there is no automatic central coordinator that assigns addresses or routes messages.

Each node must be configured with:

- Node ID
- PAN ID
- Radio channel
- Transmit power

Example command from the README:

    commission 0x0001 0x01ff 0 0

Meaning:

- Node ID = 0x0001
- PAN ID = 0x01ff
- Channel = 0
- Power = 0

Devices can communicate if:

- They are on the same PAN ID
- They are on the same channel
- They are within radio range
- They know the destination node ID

### Why radio matters here

The portable app builds a complete 90-byte message:

    message[90]

This message contains:

    network header[15] + data packet[75]

The Simplicity radio layer will later send this message using the Connect/Ember API.

The portable code should not call `emberMessageSend()` directly.

Instead:

    CookieApp_BuildDataMessage()
            ↓
    message[90]
            ↓
    Simplicity adapter
            ↓
    emberMessageSend(...)

### Radio in this project

Used for:

    Cookie → Cookie / coordinator

Important:

Radio communication is different from UART/I2C/SPI.

It is the external wireless communication layer.

## Debug / serial console

### What it is

The debug/serial console is the connection used to see logs on the PC.

In the current bring-up test, logs were viewed using:

    screen /dev/ttyACM0 115200

Observed output:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake check executed

### Why it matters here

This confirms that:

- The program is running on the Cookie.
- Custom code is being executed.
- `app_log_info()` output is visible.
- The flash/programming process works.

This is not part of the final sensor/radio system, but it is essential for debugging.

## Layer map

The intended system can be understood as:

    GNSS hardware
        ↓ UART
    Simplicity GNSS adapter
        ↓
    CookieApp_ProcessGnssEpoch()

    IMU hardware
        ↓ I2C/SPI/driver
    Simplicity IMU adapter
        ↓
    CookieApp_ProcessImuSample()

    CookieApp
        ↓
    Navigation
        ↓
    Packet builder
        ↓
    Network frame builder
        ↓
    message[90]

    Simplicity radio adapter
        ↓ Connect / Ember
    Wireless transmission

## What each project/folder is for

### `~/cookies/simplicity/Inartrans_porting/`

This is the main clean implementation.

It contains:

- Source code
- Tests
- Documentation

This is where the portable architecture is developed.

### `~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/`

This is the current Simplicity bring-up project.

It was copied from a Silicon Labs Connect Direct Mode Device example.

It is useful because it already provides:

- Simplicity v5 project structure
- Build system
- Flashing support
- Connect radio stack
- Ember APIs
- Serial logging

It is not yet the final cleaned project.

## What from `sensornode` is useful

Useful references:

- `main.c`: Simplicity v5 startup structure
- `app_init.c`: app initialisation entry point
- `app_process.c`: radio callbacks and tick callback
- `app_log_info()`: logging output
- `emberNetworkInit()`: network initialisation
- `emberMessageSend(...)`: radio send function
- `emberAfIncomingMessageCallback(...)`: receive callback
- `emberAfMessageSentCallback(...)`: send completion callback
- `emberAfStackStatusCallback(...)`: network status callback

## What from `sensornode` should not be reused blindly

The following are considered example/experimental logic and should not be copied into the final design without review:

- `sim.c`
- `sim2.c`
- `q_learning.c`
- `bucle()`
- automatic repeated transmissions
- tx test logic
- security commands
- energy scan commands
- CLI helper commands
- old experimental ACK handling

During bring-up, `bucle()` was disabled because it continuously sent messages and generated repeated:

    tag: 0, no ack received

This was unrelated to the new Inartrans_porting code.

## What from the old Inartrans code is useful

The old files should be used as functional references:

- `main`
- `flex-callbacks`
- `gps-uart`
- `imu`
- `protocolored`

They help answer:

- Which UART/baudrate was used for GNSS?
- How was the IMU initialised?
- What axis mapping was used before EKF prediction?
- What was the packet layout?
- What network state variables existed?
- How did discovery/request/repair/config work?

They should not be copied directly unless a specific responsibility has been identified.

## What has already been replaced by portable modules

The following old responsibilities already have new portable modules:

| Old responsibility | New module |
|---|---|
| GNSS parsing | `src/gnss` |
| GNSS coordinate conversion | `src/gnss` |
| IMU sample representation | `src/sensors` |
| IMU unit conversion | `src/sensors` |
| IMU axis preprocessing | `src/sensors` |
| EKF access from app | `src/navigation` |
| `paquete[75]` construction | `src/packets` |
| `header[15] + packet[75]` construction | `src/network` |
| Main data flow orchestration | `src/app` |

## What has not been replaced yet

Still pending:

- Real GNSS UART input
- Real IMU driver integration
- Real timestamps from Simplicity
- Real EKF integration
- Real radio send
- Runtime node ID / PAN ID / rank management
- Discovery/request/repair/config logic
- Standby logic
- GNSS mode runtime control
- CLI/configuration commands

## Current recommended strategy

Do not try to understand or reuse every file from the Simplicity example.

Use the Simplicity project only as a controlled bring-up environment.

Recommended order:

1. Keep the portable code clean in the repo.
2. Use the Simplicity project as a laboratory.
3. Add only small, controlled pieces.
4. Disable unrelated example logic.
5. Run fake app flow inside Simplicity.
6. Then connect one real hardware input at a time.

## Next technical step

The next technical step is to copy the portable modules into the Simplicity approach project and run a fake app flow on the real Cookie hardware.

The fake flow should do:

    Fake NMEA epoch
            ↓
    Fake IMU samples
            ↓
    CookieApp
            ↓
    message[90]
            ↓
    app_log_info confirmation

No real GNSS, IMU or radio transmission should be used yet.

____________________________________________________
____________________________________________________

# Mapa de componentes de Simplicity

## Objetivo

Este documento explica el papel de Simplicity Studio en el proyecto Inartrans_porting y ordena las distintas capas de comunicación hardware/software implicadas.

El objetivo es evitar confundir varios tipos de comunicación diferentes:

- Comunicación GNSS
- Comunicación IMU
- Comunicación radio entre Cookies
- Comunicación de debug/log con el PC

Son interfaces distintas y resuelven problemas distintos.

## Situación actual

El código portable ya se ha desarrollado y probado localmente en el repositorio:

    cookies/simplicity/Inartrans_porting/

Este código portable incluye:

- Parser y converter GNSS
- IMU sample, converter y preprocessor
- Navigation wrapper
- Packet builder
- Network frame builder
- Orquestación de app

Este código se ejecuta en tests locales de PC usando entradas falsas y un EKF mock.

El siguiente paso es integrarlo en un proyecto real de Simplicity Studio para que pueda ejecutarse en el hardware real de la Cookie.

## Dos mundos diferentes

Actualmente hay dos zonas de trabajo separadas.

### 1. Repositorio de código portable

Ubicado en:

    ~/cookies/simplicity/Inartrans_porting/

Contiene el código limpio, modular y testeable.

Debe mantenerse lo más independiente posible del hardware.

### 2. Proyecto del workspace de Simplicity Studio

Ubicado en:

    ~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/

Es un proyecto real de Simplicity Studio v5 copiado de un ejemplo de Silicon Labs Connect Direct Mode Device.

Compila, se flashea y se ejecuta en el hardware de la Cookie.

Actualmente se usa como laboratorio de bring-up, no como proyecto final limpio.

## Qué se ha verificado ya

El proyecto de Simplicity se ha probado correctamente con una comprobación fake mínima.

Verificado:

- El proyecto compila con 0 errores.
- El archivo `.hex` generado se puede flashear en la Cookie.
- La salida serial puede verse usando `screen`.
- Se puede llamar a código propio desde `ap_init()`.
- La salida de `app_log_info()` es visible.
- Se ha desactivado el loop experimental antiguo para evitar transmisiones fallidas continuas.

Salida serial observada:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake check executed
    >

Esto confirma que el proyecto actual de Simplicity es un punto de partida válido para experimentos de integración.

## Idea importante: no toda comunicación es la misma

El proyecto usa varias interfaces de comunicación, pero no son intercambiables.

Cada una conecta partes distintas del sistema.

    Módulo GNSS  → MCU de la Cookie      mediante UART
    Sensor IMU   → MCU de la Cookie      mediante I2C/SPI o driver
    MCU Cookie   → otra Cookie           mediante radio / Connect / Ember
    MCU Cookie   → terminal del PC       mediante UART / virtual COM / consola debug

Por tanto, la pregunta no es:

    ¿Debe el proyecto usar UART, I2C, SPI o radio?

La pregunta correcta es:

    ¿Qué interfaz se usa en cada frontera del sistema?

## Interfaces de comunicación

## UART

### Qué es UART

UART significa Universal Asynchronous Receiver/Transmitter.

Es una interfaz sencilla de comunicación serie.

Normalmente usa dos líneas principales:

- TX: transmisión
- RX: recepción

UART envía bytes secuencialmente, uno detrás de otro.

Se usa habitualmente para:

- Módulos GPS/GNSS
- Terminales serie de debug
- Comunicación entre microcontrolador y PC
- Módulos periféricos sencillos

### Por qué importa UART aquí

El módulo GNSS produce sentencias NMEA en texto.

Ejemplo:

    $GNRMC,123519,A,4807.038,N,01131.000,E,...

Esto es texto, enviado byte a byte.

La Cookie recibe este texto GNSS mediante una interfaz UART.

El proyecto antiguo usaba código UART en `gps-uart.c`.

El nuevo código portable no lee directamente de la UART. En su lugar, espera una época NMEA completa:

    CookieApp_ProcessGnssEpoch(buffer, length)

Por tanto, el adaptador específico de Simplicity debe:

1. Recibir caracteres desde la UART del GNSS.
2. Guardarlos en un buffer.
3. Detectar cuándo ha llegado una época GNSS completa.
4. Pasar ese texto a la app portable.

### UART en este proyecto

Se usa para:

    Módulo GNSS → Cookie

Posiblemente también para:

    Cookie → consola de debug del PC

Importante:

UART no es el enlace radio entre Cookies.

## I2C

### Qué es I2C

I2C significa Inter-Integrated Circuit.

Es un bus de comunicación usado habitualmente para conectar sensores a un microcontrolador.

Normalmente usa dos líneas:

- SDA: datos
- SCL: reloj

Varios dispositivos pueden compartir el mismo bus I2C, cada uno con una dirección distinta.

### Por qué importa I2C aquí

Muchos sensores IMU se comunican mediante I2C.

La placa Cookie puede usar un driver IMU que internamente use I2C para leer aceleración y giroscopio.

El código portable no debería depender de si la IMU usa internamente I2C o SPI.

Solo espera valores IMU lógicos:

    accel_mg[3]
    gyro_dps[3]
    timestamp_ms

### I2C en este proyecto

Puede usarse para:

    Sensor IMU → Cookie

Importante:

I2C no se usa para comunicación entre Cookies.

Normalmente se usa dentro del dispositivo, entre el microcontrolador y los sensores.

## SPI

### Qué es SPI

SPI significa Serial Peripheral Interface.

Es otra interfaz de comunicación habitual entre un microcontrolador y periféricos.

Normalmente usa:

- MOSI: master out, slave in
- MISO: master in, slave out
- SCLK: reloj
- CS: selección de chip

SPI suele ser más rápido que I2C, pero normalmente necesita más cables.

### Por qué importa SPI aquí

Algunos sensores IMU pueden comunicarse usando SPI en lugar de I2C.

Que la IMU de la Cookie use I2C o SPI depende de la placa y de la configuración del driver.

Para el código portable, este detalle debería estar oculto.

El adaptador de Simplicity o el driver IMU debe proporcionar muestras a la app en el formato esperado.

### SPI en este proyecto

Puede usarse para:

    Sensor IMU → Cookie

Importante:

Los módulos IMU portables no necesitan saber si el sensor físico usa SPI o I2C.

## Radio / Connect / Ember

### Qué es el stack de radio

La Cookie se comunica inalámbricamente con otras Cookies o con un coordinador usando el stack de radio de Silicon Labs.

El ejemplo actual de Simplicity está basado en:

    Connect: Direct Mode Device

Esto proporciona APIs como:

    emberNetworkInit()
    emberMessageSend(...)
    emberAfIncomingMessageCallback(...)
    emberAfMessageSentCallback(...)
    emberAfStackStatusCallback(...)

### Qué significa Direct Mode

Direct Mode significa que no hay un coordinador central automático que asigne direcciones o enrute mensajes.

Cada nodo debe configurarse con:

- Node ID
- PAN ID
- Canal de radio
- Potencia de transmisión

Comando de ejemplo del README:

    commission 0x0001 0x01ff 0 0

Significa:

- Node ID = 0x0001
- PAN ID = 0x01ff
- Canal = 0
- Potencia = 0

Los dispositivos pueden comunicarse si:

- Están en el mismo PAN ID
- Están en el mismo canal
- Están dentro del alcance de radio
- Conocen el node ID destino

### Por qué importa la radio aquí

La app portable construye un mensaje completo de 90 bytes:

    message[90]

Este mensaje contiene:

    cabecera de red[15] + paquete de datos[75]

La capa de radio de Simplicity enviará más adelante este mensaje usando la API Connect/Ember.

El código portable no debe llamar directamente a `emberMessageSend()`.

En su lugar:

    CookieApp_BuildDataMessage()
            ↓
    message[90]
            ↓
    Adaptador de Simplicity
            ↓
    emberMessageSend(...)

### Radio en este proyecto

Se usa para:

    Cookie → Cookie / coordinador

Importante:

La comunicación radio es distinta de UART/I2C/SPI.

Es la capa inalámbrica externa.

## Consola debug / serial

### Qué es

La consola debug/serial es la conexión usada para ver logs en el PC.

En la prueba de bring-up actual, los logs se vieron usando:

    screen /dev/ttyACM0 115200

Salida observada:

    Direct Mode Device
    Network up
    Inartrans porting app init
    Cookie porting fake check executed

### Por qué importa aquí

Esto confirma que:

- El programa se está ejecutando en la Cookie.
- El código propio se está ejecutando.
- La salida de `app_log_info()` es visible.
- El proceso de flasheo/programación funciona.

No forma parte del sistema final de sensores/radio, pero es esencial para depurar.

## Mapa de capas

El sistema previsto puede entenderse como:

    Hardware GNSS
        ↓ UART
    Adaptador GNSS de Simplicity
        ↓
    CookieApp_ProcessGnssEpoch()

    Hardware IMU
        ↓ I2C/SPI/driver
    Adaptador IMU de Simplicity
        ↓
    CookieApp_ProcessImuSample()

    CookieApp
        ↓
    Navigation
        ↓
    Packet builder
        ↓
    Network frame builder
        ↓
    message[90]

    Adaptador radio de Simplicity
        ↓ Connect / Ember
    Transmisión inalámbrica

## Para qué sirve cada proyecto/carpeta

### `~/cookies/simplicity/Inartrans_porting/`

Es la implementación limpia principal.

Contiene:

- Código fuente
- Tests
- Documentación

Aquí se desarrolla la arquitectura portable.

### `~/SimplicityStudio/v5_workspace/inartrans_porting_simplicity_approach/`

Es el proyecto actual de bring-up en Simplicity.

Se copió desde un ejemplo de Silicon Labs Connect Direct Mode Device.

Es útil porque ya proporciona:

- Estructura de proyecto Simplicity v5
- Sistema de build
- Soporte de flasheo
- Stack de radio Connect
- APIs Ember
- Logging serial

Todavía no es el proyecto final limpio.

## Qué de `sensornode` es útil

Referencias útiles:

- `main.c`: estructura de arranque de Simplicity v5
- `app_init.c`: punto de entrada de inicialización de app
- `app_process.c`: callbacks de radio y tick callback
- `app_log_info()`: salida de log
- `emberNetworkInit()`: inicialización de red
- `emberMessageSend(...)`: función de envío por radio
- `emberAfIncomingMessageCallback(...)`: callback de recepción
- `emberAfMessageSentCallback(...)`: callback de envío completado
- `emberAfStackStatusCallback(...)`: callback de estado de red

## Qué de `sensornode` NO debe reutilizarse sin revisar

Lo siguiente se considera lógica de ejemplo/experimental y no debería copiarse al diseño final sin revisión:

- `sim.c`
- `sim2.c`
- `q_learning.c`
- `bucle()`
- transmisiones automáticas repetidas
- lógica de tx test
- comandos de seguridad
- comandos de energy scan
- comandos CLI auxiliares
- gestión experimental de ACKs

Durante el bring-up, `bucle()` se desactivó porque enviaba mensajes continuamente y generaba repetidamente:

    tag: 0, no ack received

Esto no estaba relacionado con el nuevo código de Inartrans_porting.

## Qué del código antiguo de Inartrans es útil

Los archivos antiguos deben usarse como referencia funcional:

- `main`
- `flex-callbacks`
- `gps-uart`
- `imu`
- `protocolored`

Ayudan a responder:

- ¿Qué UART/baudrate se usaba para GNSS?
- ¿Cómo se inicializaba la IMU?
- ¿Qué mapeo de ejes se aplicaba antes de la predicción EKF?
- ¿Cuál era el layout del paquete?
- ¿Qué variables de estado de red existían?
- ¿Cómo funcionaban discovery/request/repair/config?

No deben copiarse directamente salvo que se haya identificado una responsabilidad concreta.

## Qué ya ha sido sustituido por módulos portables

Las siguientes responsabilidades antiguas ya tienen nuevos módulos portables:

| Responsabilidad antigua | Módulo nuevo |
|---|---|
| Parseo GNSS | `src/gnss` |
| Conversión de coordenadas GNSS | `src/gnss` |
| Representación de muestra IMU | `src/sensors` |
| Conversión de unidades IMU | `src/sensors` |
| Preprocesado de ejes IMU | `src/sensors` |
| Acceso al EKF desde la app | `src/navigation` |
| Construcción de `paquete[75]` | `src/packets` |
| Construcción de `header[15] + packet[75]` | `src/network` |
| Orquestación principal del flujo de datos | `src/app` |

## Qué no ha sido sustituido todavía

Pendiente:

- Entrada UART real del GNSS
- Integración con driver IMU real
- Timestamps reales desde Simplicity
- Integración con EKF real
- Envío real por radio
- Gestión runtime de node ID / PAN ID / rango
- Lógica discovery/request/repair/config
- Lógica de standby
- Control runtime de modo GNSS
- Comandos CLI/configuración

## Estrategia recomendada actual

No intentar entender o reutilizar todos los archivos del ejemplo de Simplicity.

Usar el proyecto de Simplicity solo como entorno controlado de bring-up.

Orden recomendado:

1. Mantener el código portable limpio en el repo.
2. Usar el proyecto de Simplicity como laboratorio.
3. Añadir solo piezas pequeñas y controladas.
4. Desactivar lógica de ejemplo no relacionada.
5. Ejecutar fake app flow dentro de Simplicity.
6. Después conectar una entrada hardware real cada vez.

## Próximo paso técnico

El próximo paso técnico es copiar los módulos portables al proyecto approach de Simplicity y ejecutar un fake app flow en el hardware real de la Cookie.

El fake flow debe hacer:

    Época NMEA falsa
            ↓
    Muestras IMU falsas
            ↓
    CookieApp
            ↓
    message[90]
            ↓
    confirmación con app_log_info

Todavía no deben usarse GNSS real, IMU real ni transmisión radio real.