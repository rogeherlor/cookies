# Old System Overview

This document describes the structure and behaviour of the original cookie firmware.

The purpose is not to fully understand every detail, but to identify the main functional blocks and data flow.

## High-Level Flow

The system operates as a combination of three main processes:

- IMU sampling (high frequency)
- GNSS processing (low frequency)
- Data transmission over the network (periodic)

These processes are coordinated through the main application loop.

## Main Execution Points

The core of the system is built around three key functions:

- `emberAfMainInitCallback()` → system initialization
- `emberAfMainTickCallback()` → continuous execution loop
- `reportHandler()` → periodic data transmission

## Functional Blocks

### 1. IMU Handling

- Data is sampled at ~200 Hz using a timer.
- Accelerometer and gyroscope data are read and stored.
- Every two samples, EKF prediction is executed.

Relevant variables:
- `last_acelint[]`
- `last_gyroint[]`

---

### 2. GNSS Handling

- GNSS data is received as NMEA sentences via UART.
- A full epoch is detected when multiple lines are received.
- Data is parsed into usable variables.

Relevant variables:
- `latitud_s`
- `longitud_s`
- `altitud_s`
- `vel_GNSS_u`
- `PDOP_u`

---

### 3. EKF (Navigation)

- EKF prediction runs using IMU data.
- EKF update runs when valid GNSS data is available.
- The filter outputs position and velocity estimates.

Main functions:
- `EKF_Init()`
- `EKF_Predict()`
- `EKF_Update()`

---

### 4. Packet Construction

- A fixed-size packet (75 bytes) is built.
- Data includes IMU, GNSS, and EKF outputs.
- Memory is filled using `memcpy()` with fixed offsets.

---

### 5. Network Communication

- Nodes form a tree structure using "parent" and "rank".
- Data is sent periodically to a parent node.
- Intermediate nodes forward packets.
- The coordinator receives and processes all data.

Key concepts:
- Parent node selection
- RSSI-based decisions
- Multi-hop forwarding

---

### 6. Event and Callback System

The system relies heavily on callbacks:

- Incoming messages
- Message sent status
- Network state changes

These callbacks handle communication and network maintenance.

---

## Key Observations

- The system is functionally complete but highly coupled.
- Responsibilities are mixed within large functions.
- Data flow is implicit rather than structured.
- Packet construction relies on hardcoded offsets.
- GNSS parsing and EKF integration are intertwined with application logic.

---

## Goal for Refactoring

The new implementation should:

- Separate each functional block into independent modules.
- Make data flow explicit and easier to follow.
- Isolate hardware-specific code from application logic.
- Improve readability and maintainability.

____________________________________________________
____________________________________________________


# Mapa del sistema antiguo

Este documento describe la estructura y el comportamiento del firmware antiguo de las cookies.

El objetivo no es entender cada línea de código, sino identificar los bloques funcionales principales y el flujo de datos.

## Flujo general

El sistema combina tres procesos principales:

- Muestreo de la IMU, a alta frecuencia.
- Procesamiento GNSS, a baja frecuencia.
- Envío periódico de datos por la red.

Estos procesos se coordinan desde el bucle principal de la aplicación.

## Puntos principales de ejecución

El sistema se apoya sobre tres funciones clave:

- `emberAfMainInitCallback()` → inicialización del sistema.
- `emberAfMainTickCallback()` → ejecución continua.
- `reportHandler()` → envío periódico de datos.

## Bloques funcionales

### 1. Gestión de la IMU

- La IMU se muestrea a unos 200 Hz usando un temporizador.
- Se leen datos del acelerómetro y del giróscopo.
- Cada dos muestras, se ejecuta la predicción del EKF.

Variables relevantes:

- `last_acelint[]`
- `last_gyroint[]`

### 2. Gestión del GNSS

- Los datos GNSS llegan por UART como sentencias NMEA.
- Se detecta una época completa cuando han llegado varias líneas.
- Las sentencias se parsean para obtener variables útiles.

Variables relevantes:

- `latitud_s`
- `longitud_s`
- `altitud_s`
- `vel_GNSS_u`
- `PDOP_u`

### 3. EKF / navegación

- La predicción del EKF se ejecuta usando los datos de la IMU.
- La actualización del EKF se ejecuta cuando hay datos GNSS válidos.
- El filtro proporciona estimaciones de posición y velocidad.

Funciones principales:

- `EKF_Init()`
- `EKF_Predict()`
- `EKF_Update()`

### 4. Construcción del paquete

- Se construye un paquete de tamaño fijo de 75 bytes.
- El paquete incluye datos de IMU, GNSS y EKF.
- Los campos se colocan usando `memcpy()` con posiciones fijas.

### 5. Comunicación por red

- Los nodos forman una estructura de árbol usando “padre” y “rango”.
- Cada nodo envía datos periódicamente hacia su nodo padre.
- Los nodos intermedios reenvían paquetes.
- El coordinador recibe y procesa los datos finales.

Conceptos clave:

- Selección de nodo padre.
- Decisiones basadas en RSSI.
- Reenvío multi-salto.

### 6. Sistema de eventos y callbacks

El sistema depende mucho de callbacks:

- Recepción de mensajes.
- Estado del envío de mensajes.
- Cambios de estado de red.

Estos callbacks gestionan la comunicación y el mantenimiento de la red.

## Observaciones principales

- El sistema funciona, pero está muy acoplado.
- Las responsabilidades están mezcladas en funciones largas.
- El flujo de datos no está claramente estructurado.
- La construcción del paquete depende de offsets escritos a mano.
- El parseo GNSS y el EKF están mezclados con la lógica general de la aplicación.

## Objetivo de la refactorización

La nueva implementación debería:

- Separar cada bloque funcional en módulos independientes.
- Hacer explícito el flujo de datos.
- Aislar el código específico de hardware de la lógica de aplicación.
- Mejorar la legibilidad y el mantenimiento.