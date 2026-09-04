# App Orchestration

## Purpose

The app orchestration module coordinates the portable software flow before integrating it into Simplicity Studio.

It connects GNSS input, IMU input, navigation, packet generation and network frame generation through a clean application-level interface.

The goal is to keep the main application logic independent from hardware, UART, timers, radio callbacks and Simplicity-specific APIs.

## What it does

The app module:

- Processes GNSS epochs
- Processes IMU samples
- Updates the navigation module
- Keeps the latest GNSS, IMU and navigation state
- Builds the 75-byte data packet
- Builds the 90-byte network message

## Structure

    GNSS epoch
            ↓
    CookieApp_ProcessGnssEpoch()

    IMU sample
            ↓
    CookieApp_ProcessImuSample()

    Internal app state
            ↓
    CookieApp_BuildDataMessage()
            ↓
    message[90]

## Main files

    src/app/app.h
    src/app/app.c

## Main types

### CookieAppConfig

Stores configuration used by the app.

It currently contains:

- GNSS mode
- Network frame header

The GNSS mode is still a placeholder in the local test. In the real application, it should come from the GNSS configuration/state logic.

The network header contains the metadata required to build the 90-byte message.

### CookieAppContext

Stores the runtime state of the app.

It contains:

- Last GNSS fix
- Last IMU sample
- Last navigation state
- Flags indicating whether GNSS, IMU and navigation data are available
- App configuration

## Main functions

### CookieApp_Init

Initialises the app context.

It also resets the navigation module and the IMU converter.

### CookieApp_ProcessGnssEpoch

Processes a complete NMEA GNSS epoch.

It:

1. Parses the GNSS epoch.
2. Stores the latest valid fix.
3. Updates the navigation module with GNSS data.
4. Updates the cached navigation state when available.

### CookieApp_ProcessImuSample

Processes one IMU sample.

It:

1. Stores the raw IMU sample.
2. Converts raw IMU data into physical units.
3. Ignores the first sample for EKF prediction if `dt = 0`.
4. Applies IMU preprocessing for navigation.
5. Calls the navigation prediction step when navigation is initialised.
6. Updates the cached navigation state.

### CookieApp_BuildDataMessage

Builds the final 90-byte message.

It:

1. Creates a `CookiePacketData` structure from the app state.
2. Builds the legacy 75-byte data packet.
3. Wraps it with the 15-byte network header.
4. Produces a complete `message[90]`.

## Test

The local test verifies the app flow without hardware.

Test file:

    tests/test_app_flow.c

Execution:

    ./simplicity/Inartrans_porting/tests/test_app_flow

## Test flow

    Fake GNSS epoch
            ↓
    CookieApp_ProcessGnssEpoch()
            ↓
    Navigation update / mock EKF

    Fake IMU samples
            ↓
    CookieApp_ProcessImuSample()
            ↓
    IMU conversion
            ↓
    IMU preprocessing
            ↓
    Navigation prediction / mock EKF

    App state
            ↓
    CookieApp_BuildDataMessage()
            ↓
    packet[75]
            ↓
    message[90]

## Expected behaviour

- GNSS epochs are parsed correctly.
- Navigation waits for several valid GNSS fixes before becoming available.
- The first IMU sample initializes the converter timestamp.
- The second IMU sample provides a positive `dt`.
- Navigation prediction is executed.
- The app builds a complete 90-byte message.
- Key legacy fields are correctly preserved:
  - packet type = `3`
  - payload validity byte = `A`
  - GNSS mode byte = `7`
  - final legacy markers = `,,`

## What it does NOT validate

This test does not validate:

- Real GNSS UART input
- Real IMU hardware
- Real EKF behaviour
- Physical navigation accuracy
- Simplicity Studio integration
- Radio transmission
- ACK handling
- Routing logic
- Discovery/request/repair/config behaviour

## Design rationale

The original implementation mixed:

- Sensor reading
- GNSS parsing
- IMU processing
- EKF calls
- Packet generation
- Network framing
- Radio transmission

The app orchestration module separates the portable application logic from hardware-specific code.

This makes it possible to test most of the flow locally before integrating with Simplicity Studio.

## Role in future Simplicity integration

In the future Simplicity version:

- UART/GNSS code will provide NMEA epochs to `CookieApp_ProcessGnssEpoch()`.
- IMU driver/timer code will provide samples to `CookieApp_ProcessImuSample()`.
- Radio/network code will send the `message[90]` produced by `CookieApp_BuildDataMessage()`.

Simplicity should act mainly as the hardware and runtime adapter.

## Current status

Implemented:

- Portable app context
- GNSS processing entry point
- IMU processing entry point
- Message generation entry point
- Local app flow test

Not implemented yet:

- Simplicity callbacks
- UART connection
- IMU hardware connection
- Real radio send
- Runtime network state management
- Discovery/request/repair/config handling

____________________________________________________
____________________________________________________

# Orquestación de la app

## Objetivo

El módulo de orquestación de la app coordina el flujo software portable antes de integrarlo en Simplicity Studio.

Conecta entrada GNSS, entrada IMU, navegación, generación de paquete y generación de trama de red mediante una interfaz limpia a nivel de aplicación.

El objetivo es mantener la lógica principal independiente del hardware, UART, timers, callbacks de radio y APIs específicas de Simplicity.

## Qué hace

El módulo app:

- Procesa épocas GNSS
- Procesa muestras IMU
- Actualiza el módulo de navegación
- Mantiene el último estado GNSS, IMU y navigation
- Construye el paquete de datos de 75 bytes
- Construye el mensaje de red de 90 bytes

## Estructura

    Época GNSS
            ↓
    CookieApp_ProcessGnssEpoch()

    Muestra IMU
            ↓
    CookieApp_ProcessImuSample()

    Estado interno de app
            ↓
    CookieApp_BuildDataMessage()
            ↓
    message[90]

## Archivos principales

    src/app/app.h
    src/app/app.c

## Tipos principales

### CookieAppConfig

Almacena la configuración utilizada por la app.

Actualmente contiene:

- Modo GNSS
- Cabecera de trama de red

El modo GNSS sigue siendo un placeholder en el test local. En la aplicación real deberá venir de la lógica de configuración/estado del GNSS.

La cabecera de red contiene los metadatos necesarios para construir el mensaje de 90 bytes.

### CookieAppContext

Almacena el estado runtime de la app.

Contiene:

- Último fix GNSS
- Última muestra IMU
- Último estado de navegación
- Flags que indican si hay datos GNSS, IMU y navegación disponibles
- Configuración de la app

## Funciones principales

### CookieApp_Init

Inicializa el contexto de la app.

También resetea el módulo de navegación y el converter de IMU.

### CookieApp_ProcessGnssEpoch

Procesa una época GNSS NMEA completa.

Hace:

1. Parsea la época GNSS.
2. Guarda el último fix válido.
3. Actualiza el módulo de navegación con datos GNSS.
4. Actualiza el estado de navegación cacheado cuando está disponible.

### CookieApp_ProcessImuSample

Procesa una muestra IMU.

Hace:

1. Guarda la muestra IMU cruda.
2. Convierte los datos IMU crudos a unidades físicas.
3. Ignora la primera muestra para predicción EKF si `dt = 0`.
4. Aplica el preprocesado IMU para navegación.
5. Llama a la predicción de navegación cuando navigation está inicializado.
6. Actualiza el estado de navegación cacheado.

### CookieApp_BuildDataMessage

Construye el mensaje final de 90 bytes.

Hace:

1. Crea una estructura `CookiePacketData` a partir del estado de la app.
2. Construye el paquete legacy de 75 bytes.
3. Lo envuelve con la cabecera de red de 15 bytes.
4. Genera un `message[90]` completo.

## Test

El test local verifica el flujo de la app sin hardware.

Archivo de test:

    tests/test_app_flow.c

Ejecución:

    ./simplicity/Inartrans_porting/tests/test_app_flow

## Flujo del test

    Época GNSS falsa
            ↓
    CookieApp_ProcessGnssEpoch()
            ↓
    Actualización de navegación / EKF mock

    Muestras IMU falsas
            ↓
    CookieApp_ProcessImuSample()
            ↓
    Conversión IMU
            ↓
    Preprocesado IMU
            ↓
    Predicción de navegación / EKF mock

    Estado de app
            ↓
    CookieApp_BuildDataMessage()
            ↓
    packet[75]
            ↓
    message[90]

## Comportamiento esperado

- Las épocas GNSS se parsean correctamente.
- Navigation espera varios fixes GNSS válidos antes de estar disponible.
- La primera muestra IMU inicializa el timestamp del converter.
- La segunda muestra IMU proporciona un `dt` positivo.
- Se ejecuta la predicción de navegación.
- La app construye un mensaje completo de 90 bytes.
- Los campos legacy principales se mantienen correctamente:
  - tipo de paquete = `3`
  - byte de validez del payload = `A`
  - byte de modo GNSS = `7`
  - marcadores legacy finales = `,,`

## Qué NO valida

Este test no valida:

- Entrada GNSS real por UART
- Hardware IMU real
- Comportamiento del EKF real
- Precisión física de navegación
- Integración con Simplicity Studio
- Transmisión por radio
- Gestión de ACKs
- Lógica de routing
- Comportamiento discovery/request/repair/config

## Razonamiento de diseño

La implementación original mezclaba:

- Lectura de sensores
- Parseo GNSS
- Procesado IMU
- Llamadas al EKF
- Generación de paquete
- Construcción de trama de red
- Transmisión por radio

El módulo de orquestación de app separa la lógica portable de aplicación del código específico de hardware.

Esto permite probar la mayor parte del flujo localmente antes de integrarlo con Simplicity Studio.

## Rol en la futura integración con Simplicity

En la futura versión de Simplicity:

- El código UART/GNSS entregará épocas NMEA a `CookieApp_ProcessGnssEpoch()`.
- El driver/timer de IMU entregará muestras a `CookieApp_ProcessImuSample()`.
- El código de radio/red enviará el `message[90]` generado por `CookieApp_BuildDataMessage()`.

Simplicity debería actuar principalmente como adaptador de hardware y runtime.

## Estado actual

Implementado:

- Contexto portable de app
- Punto de entrada para procesado GNSS
- Punto de entrada para procesado IMU
- Punto de entrada para generación de mensaje
- Test local del flujo de app

No implementado aún:

- Callbacks de Simplicity
- Conexión UART
- Conexión con hardware IMU
- Envío real por radio
- Gestión runtime del estado de red
- Manejo de discovery/request/repair/config