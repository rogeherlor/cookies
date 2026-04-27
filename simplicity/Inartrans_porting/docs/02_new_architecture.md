# Proposed New Architecture

This document describes the proposed architecture for `Inartrans_porting`.

The goal is to rewrite the system in a modular way, using the old code as a functional reference, but avoiding its current structure.

## Core idea

The cookie should follow this data flow:

IMU + GNSS  
↓  
Navigation / EKF  
↓  
Packet construction  
↓  
Network communication  

The system must preserve the functionality of the original code, but with clearly separated responsibilities and a cleaner data flow.

## Proposed structure

Inartrans_porting/
├── docs/
├── src/
│   ├── app/
│   ├── config/
│   ├── sensors/
│   ├── gnss/
│   ├── navigation/
│   ├── packets/
│   ├── network/
│   └── cli/

## Module responsibilities

### app/

Contains the main application logic.

Responsibilities:

- Initialise modules.
- Coordinate the overall flow.
- Execute periodic tasks.
- Call sensors, navigation, packets, and network modules.

Expected functions:

- App_Init()
- App_Tick()

---

### config/

Contains global configuration parameters.

Examples:

- Node mode: sensor or coordinator.
- Transmission period.
- Default GNSS mode.
- Packet size.
- Sampling frequencies.

---

### sensors/

Handles physical sensors, mainly the IMU.

Responsibilities:

- Initialise the IMU.
- Read acceleration.
- Read gyroscope.
- Store the latest available sample.

Example data structure:

typedef struct {
    int32_t accel_mg[3];
    int32_t gyro_dps[3];
    uint32_t timestamp_ms;
} CookieImuSample;

---

### gnss/

Handles GNSS logic.

Responsibilities:

- Read GNSS data via UART.
- Accumulate NMEA sentences.
- Parse the sentences.
- Generate a clean GNSS fix.
- Manage GNSS modes if required.

Example data structure:

typedef struct {
    bool valid;
    float latitude;
    float longitude;
    float altitude;
    uint16_t speed_cm_s;
    uint16_t pdop_x100;
    char time_utc[11];
    char date_ymd[7];
} CookieGnssFix;

---

### navigation/

Handles integration with the existing EKF.

Responsibilities:

- Initialise the EKF when valid GNSS is available.
- Run prediction using IMU data.
- Run update using GNSS data.
- Provide filtered position.

The EKF is not reimplemented. The existing code is reused.

Example data structure:

typedef struct {
    bool valid;
    float latitude;
    float longitude;
    float altitude;
    float velocity;
} CookieNavigationState;

---

### packets/

Handles packet construction and interpretation.

Responsibilities:

- Build data payloads.
- Parse received payloads.
- Centralise packet offsets.
- Avoid scattered memcpy usage.

Example:

#define COOKIE_DATA_PAYLOAD_SIZE 75

Expected functions:

- CookiePacket_EncodeData()
- CookiePacket_DecodeData()

---

### network/

Handles radio communication.

Responsibilities:

- Initialise the network.
- Manage parent, rank, and PAN ID.
- Send packets.
- Relay packets if acting as an intermediate node.
- Handle discovery, request, repair, and configuration messages.

Expected functions:

- CookieNetwork_Init()
- CookieNetwork_SendData()
- CookieNetwork_HandleIncomingMessage()

---

### cli/

Handles manual commands and configuration.

Responsibilities:

- Change transmission period.
- Enable/disable standby mode.
- Change GNSS mode.
- Display node information.

---

## Execution flow

### Initialisation

main()  
↓  
App_Init()  
↓  
CookieIMU_Init()  
CookieGNSS_Init()  
CookieNavigation_Init()  
CookieNetwork_Init()

---

### Continuous execution

main loop  
↓  
App_Tick()  
↓  
1. Process IMU if a new sample is available  
2. Process GNSS if a new epoch is available  
3. Update EKF  
4. Send packet if required  
5. Handle network events  

---

### Data transmission

latest IMU sample  
latest GNSS fix  
latest EKF state  
↓  
CookiePacket_EncodeData()  
↓  
CookieNetwork_SendData()

---

### Data reception

incoming message  
↓  
CookieNetwork_HandleIncomingMessage()  
↓  
if it is for this node:  
    CookiePacket_DecodeData()  
else:  
    relay to parent node  

---

## Design criteria

The new architecture must:

- Assign a clear responsibility to each module.
- Use structured data instead of scattered global variables.
- Centralise packet layout definitions.
- Keep Simplicity callbacks short.
- Separate hardware-specific code from application logic.
- Preserve the behaviour of the original system.

---

## Implementation strategy

The system should be developed in small milestones:

1. Create folder structure and documentation.
2. Implement minimal App_Init() and App_Tick().
3. Add IMU reading.
4. Add GNSS parsing.
5. Integrate the existing EKF.
6. Implement packet construction.
7. Add network transmission.
8. Add reception and relaying.
9. Add configuration commands.

---

## Important note

The goal is not to copy the old code with new names.  
The goal is to preserve its behaviour while achieving a cleaner, more maintainable, and portable structure.


____________________________________________________
____________________________________________________

# Nueva arquitectura propuesta

Este documento describe la arquitectura propuesta para `Inartrans_porting`.

El objetivo es reescribir el sistema de forma modular, usando el código antiguo como referencia funcional, pero evitando mantener su estructura actual.

## Idea principal

La cookie debe realizar el siguiente flujo:

IMU + GNSS  
↓  
Navegación / EKF  
↓  
Construcción de paquete  
↓  
Comunicación por red  

El sistema debe conservar la funcionalidad del código antiguo, pero con responsabilidades separadas y un flujo de datos más claro.

## Estructura propuesta

Inartrans_porting/
├── docs/
├── src/
│   ├── app/
│   ├── config/
│   ├── sensors/
│   ├── gnss/
│   ├── navigation/
│   ├── packets/
│   ├── network/
│   └── cli/

## Responsabilidad de cada módulo

### app/

Contiene la lógica principal de la aplicación.

Responsabilidades:

- Inicializar los módulos.
- Coordinar el flujo general.
- Ejecutar tareas periódicas.
- Llamar a sensores, navegación, paquetes y red.

Funciones esperadas:

- App_Init()
- App_Tick()

### config/

Contiene parámetros globales de configuración.

Ejemplos:

- Modo de nodo: sensor o coordinador.
- Periodo de envío.
- Modo GNSS por defecto.
- Tamaño del paquete.
- Frecuencias de muestreo.

### sensors/

Contiene la lógica relacionada con sensores físicos, especialmente la IMU.

Responsabilidades:

- Inicializar la IMU.
- Leer aceleración.
- Leer giróscopo.
- Guardar la última muestra disponible.

Ejemplo de dato:

typedef struct {
    int32_t accel_mg[3];
    int32_t gyro_dps[3];
    uint32_t timestamp_ms;
} CookieImuSample;

### gnss/

Contiene la lógica del GNSS.

Responsabilidades:

- Leer datos GNSS por UART.
- Acumular sentencias NMEA.
- Parsear las sentencias.
- Generar una posición GNSS limpia.
- Gestionar el modo GNSS si es necesario.

Ejemplo de dato:

typedef struct {
    bool valid;
    float latitude;
    float longitude;
    float altitude;
    uint16_t speed_cm_s;
    uint16_t pdop_x100;
    char time_utc[11];
    char date_ymd[7];
} CookieGnssFix;

### navigation/

Contiene la integración con el EKF existente.

Responsabilidades:

- Inicializar el EKF cuando haya GNSS válido.
- Ejecutar predicción con datos IMU.
- Ejecutar actualización con datos GNSS.
- Proporcionar la posición filtrada.

El EKF no se reimplementa. Se utiliza el código ya existente como dependencia.

Ejemplo de dato:

typedef struct {
    bool valid;
    float latitude;
    float longitude;
    float altitude;
    float velocity;
} CookieNavigationState;

### packets/

Contiene la construcción e interpretación de paquetes.

Responsabilidades:

- Construir el payload de datos.
- Parsear payloads recibidos.
- Centralizar los offsets del paquete.
- Evitar memcpy dispersos por todo el código.

Ejemplo:

#define COOKIE_DATA_PAYLOAD_SIZE 75

Funciones esperadas:

- CookiePacket_EncodeData()
- CookiePacket_DecodeData()

### network/

Contiene la lógica de comunicación por radio.

Responsabilidades:

- Inicializar la red.
- Gestionar padre, rango y PAN ID.
- Enviar paquetes.
- Reenviar paquetes si el nodo actúa como relay.
- Gestionar discovery, request, repair y config.

Funciones esperadas:

- CookieNetwork_Init()
- CookieNetwork_SendData()
- CookieNetwork_HandleIncomingMessage()

### cli/

Contiene comandos manuales o de configuración.

Responsabilidades:

- Cambiar periodo de envío.
- Activar o desactivar standby.
- Cambiar modo GNSS.
- Mostrar información del nodo.

## Flujo de ejecución propuesto

### Inicialización

main()  
↓  
App_Init()  
↓  
CookieIMU_Init()  
CookieGNSS_Init()  
CookieNavigation_Init()  
CookieNetwork_Init()

### Ejecución continua

main loop  
↓  
App_Tick()  
↓  
1. Procesar IMU si hay nueva muestra  
2. Procesar GNSS si hay nueva época  
3. Actualizar EKF  
4. Enviar paquete si toca  
5. Procesar eventos de red  

### Envío de datos

última muestra IMU  
último fix GNSS  
último estado EKF  
↓  
CookiePacket_EncodeData()  
↓  
CookieNetwork_SendData()

### Recepción de datos

mensaje recibido  
↓  
CookieNetwork_HandleIncomingMessage()  
↓  
si es para este nodo:  
    CookiePacket_DecodeData()  
si no:  
    reenviar al padre  

## Criterios de diseño

La nueva arquitectura debe cumplir:

- Cada módulo debe tener una responsabilidad clara.
- Los datos deben moverse mediante estructuras, no mediante variables globales dispersas.
- Las posiciones del paquete deben estar centralizadas.
- Los callbacks de Simplicity deben ser lo más cortos posible.
- El código específico de hardware debe estar separado de la lógica de aplicación.
- El comportamiento funcional debe mantenerse respecto al sistema antiguo.

## Estrategia de implementación

La implementación debe hacerse por hitos pequeños:

1. Crear estructura de carpetas y documentación.
2. Crear App_Init() y App_Tick() mínimos.
3. Añadir lectura IMU.
4. Añadir parseo GNSS.
5. Integrar el EKF existente.
6. Construir paquete de datos.
7. Añadir envío por red.
8. Añadir recepción y reenvío.
9. Añadir comandos de configuración.

## Nota importante

El objetivo no es copiar el código antiguo con nombres nuevos.  
El objetivo es conservar su comportamiento, pero con una estructura más clara, mantenible y fácil de portar a versiones actuales de Simplicity Studio.