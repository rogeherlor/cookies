# Packet Builder

## Purpose

The packet builder module generates the 75-byte data packet used for communication.

It converts structured internal data into a fixed binary format required by the existing system.

The goal is to keep the external protocol unchanged while redesigning the internal code to be clean, structured, and maintainable.

---

## What it does

The packet builder:

* Takes structured data (`CookiePacketData`)
* Places each field into a fixed position in a 75-byte buffer
* Produces a packet fully compatible with the original implementation

---

## Structure

The packet generation is divided into two layers:

```
CookiePacketData → Packet Builder → 75-byte buffer
```

---

## CookiePacketData

### Purpose

Represents all the data to be transmitted in a clean and structured way.

### Structure

```
CookiePacketData
```

Contains grouped data:

* GNSS (raw values)
* Navigation (processed values)
* IMU data
* Status / flags

### Characteristics

* Logical grouping of data
* Easy to understand and debug
* Independent from packet layout
* No knowledge of byte offsets

---

## Packet Builder

### Purpose

Transforms structured data into the legacy binary packet format.

### Input

```
CookiePacketData
```

### Output

```
uint8_t buffer[75]
```

### What it does

1. Extracts values from `CookiePacketData`
2. Places them at fixed byte offsets
3. Ensures correct packet size (75 bytes)

### What it does NOT do

* Sensor reading
* Data preprocessing
* Unit conversion
* Navigation logic
* EKF logic
* Validation of physical correctness

---

## Packet layout

The packet has a fixed size:

```
75 bytes
```

The position of each field is defined by the legacy protocol.

Important:

* Fields are **not grouped logically**
* GNSS, navigation and IMU data may be interleaved
* The layout must be preserved for compatibility

---

## Design rationale

The original implementation mixed:

* Data generation
* Packet formatting
* Byte-level operations

This module separates responsibilities to:

* Improve clarity
* Enable testing
* Avoid hardcoded logic spread across the code
* Keep compatibility with the existing system

The packet layout is intentionally preserved.

---

## Expected behaviour

* The packet always has a size of 75 bytes
* Fields are placed at the correct offsets
* Key fields (e.g., validity, GNSS mode) are correctly encoded
* The packet can be interpreted by the existing system

---

## What it does NOT validate

This module does not validate:

* Physical correctness of the data
* GNSS accuracy
* IMU correctness
* Navigation consistency
* EKF behaviour
* Communication reliability

---

## Testing

A dedicated test verifies:

* Packet size (75 bytes)
* Correct field placement
* Basic field values

Test file:

```
tests/test_packet_builder.c
```

Execution:

```
./simplicity/Inartrans_porting/tests/test_packet_builder
```

---

## Design decision

The internal structure and the packet layout are intentionally different.

* Internal representation → clean and grouped
* Packet format → fixed and legacy

The packet builder acts as a translation layer between both.

---

## Future improvements

Possible future steps:

* Define a new packet format (v2)
* Add versioning support
* Include checksum or integrity fields
* Implement a packet parser (reverse operation)

---

---

# CookiePacketData

## Purpose

The `CookiePacketData` structure represents all the data to be transmitted in a clean and structured way.

It acts as the internal data model used by the application before being converted into a binary packet.

The goal is to decouple data representation from packet formatting.

---

## What it does

`CookiePacketData`:

* Stores all relevant data for transmission
* Groups fields by meaning (GNSS, navigation, IMU, status)
* Provides a clear interface between modules and the packet builder

---

## Structure

```
CookiePacketData
```

The structure is composed of logically grouped sub-blocks:

* GNSS data (raw sensor values)
* Navigation data (processed values)
* IMU data
* Status / flags

---

## GNSS data

### Purpose

Represents raw positioning data obtained from the GNSS module.

### Characteristics

* Direct output from GNSS processing
* No additional interpretation
* Typically includes raw latitude, longitude, altitude and mode

---

## Navigation data

### Purpose

Represents processed positioning data used by the navigation system.

### Characteristics

* Derived from GNSS and IMU fusion
* Expressed in physical units (e.g., degrees, meters)
* Used as the main output of the navigation module

---

## IMU data

### Purpose

Represents inertial data to be included in the packet.

### Characteristics

* May include acceleration and angular velocity
* Already preprocessed and aligned for navigation if required
* No additional transformations inside this structure

---

## Status / flags

### Purpose

Stores system-level information related to packet validity and state.

### Examples

* Validity flag (e.g., 'A' / 'V')
* GNSS mode
* Other system indicators if needed

---

## Design principles

The structure follows these principles:

* Logical grouping of data
* Clear separation of responsibilities
* Independence from packet layout
* Simplicity and readability
* Ease of testing

---

## What it does NOT do

`CookiePacketData` does not:

* Perform packet formatting
* Handle byte-level operations
* Know packet size or offsets
* Perform unit conversion
* Read sensors
* Execute navigation or EKF logic

---

## Role in the system

`CookiePacketData` sits between:

```
Navigation / Sensors → CookiePacketData → Packet Builder → Binary packet
```

It serves as the unified data container used across modules.

---

## Example flow

```
GNSS data
        ↓
Navigation update
        ↓
IMU processing
        ↓
CookiePacketData
        ↓
Packet Builder
        ↓
75-byte packet
```

---

## Design rationale

The original implementation mixed:

* Data generation
* Packet formatting
* Byte-level manipulation

This structure isolates the data model from the packet format to:

* Improve clarity
* Enable modular design
* Simplify debugging
* Avoid tightly coupled code

---

## Expected behaviour

* Data is stored in a clear and structured way
* Fields are easy to access and modify
* The structure can be used independently of the packet builder

---

## Future improvements

Possible extensions:

* Add new fields without affecting packet layout
* Support multiple packet formats
* Introduce versioning if needed
* Extend status information

---

---

# Packet Builder

## Objetivo

El módulo packet builder genera el paquete de datos de 75 bytes utilizado en la comunicación.

Convierte datos internos estructurados en un formato binario fijo requerido por el sistema existente.

El objetivo es mantener el protocolo externo sin cambios mientras se rediseña el código interno para que sea limpio, estructurado y mantenible.

---

## Qué hace

El packet builder:

* Recibe datos estructurados (`CookiePacketData`)
* Coloca cada campo en una posición fija dentro de un buffer de 75 bytes
* Genera un paquete totalmente compatible con la implementación original

---

## Estructura

La generación del paquete se divide en dos capas:

```
CookiePacketData → Packet Builder → buffer de 75 bytes
```

---

## CookiePacketData

### Objetivo

Representa todos los datos a transmitir de forma clara y estructurada.

### Estructura

```
CookiePacketData
```

Contiene datos agrupados:

* GNSS (valores crudos)
* Navigation (valores procesados)
* Datos IMU
* Estado / flags

### Características

* Agrupación lógica de los datos
* Fácil de entender y depurar
* Independiente del layout del paquete
* Sin conocimiento de offsets de bytes

---

## Packet Builder

### Objetivo

Transforma los datos estructurados en el formato binario legacy.

### Entrada

```
CookiePacketData
```

### Salida

```
uint8_t buffer[75]
```

### Qué hace

1. Extrae valores de `CookiePacketData`
2. Los coloca en offsets fijos
3. Garantiza el tamaño del paquete (75 bytes)

### Qué NO hace

* Lectura de sensores
* Preprocesado de datos
* Conversión de unidades
* Lógica de navegación
* Lógica del EKF
* Validación física de datos

---

## Layout del paquete

El paquete tiene un tamaño fijo:

```
75 bytes
```

La posición de cada campo viene definida por el protocolo legacy.

Importante:

* Los campos **no están agrupados lógicamente**
* Datos GNSS, navegación e IMU pueden estar intercalados
* El layout debe mantenerse para compatibilidad

---

## Razonamiento de diseño

La implementación original mezclaba:

* Generación de datos
* Formateo del paquete
* Operaciones a nivel de byte

Este módulo separa responsabilidades para:

* Mejorar claridad
* Permitir testing
* Evitar lógica hardcodeada distribuida
* Mantener compatibilidad con el sistema existente

El layout del paquete se mantiene de forma intencionada.

---

## Comportamiento esperado

* El paquete siempre tiene 75 bytes
* Los campos se colocan en los offsets correctos
* Campos clave (ej. validity, GNSS mode) se codifican correctamente
* El paquete puede ser interpretado por el sistema existente

---

## Qué NO valida

Este módulo no valida:

* Corrección física de los datos
* Precisión GNSS
* Datos IMU reales
* Consistencia de navegación
* Comportamiento del EKF
* Fiabilidad de la comunicación

---

## Testing

Existe un test específico que verifica:

* Tamaño del paquete (75 bytes)
* Colocación de campos
* Valores básicos

Archivo de test:

```
tests/test_packet_builder.c
```

Ejecución:

```
./simplicity/Inartrans_porting/tests/test_packet_builder
```

---

## Decisión de diseño

La estructura interna y el layout del paquete son diferentes de forma intencionada.

* Representación interna → limpia y agrupada
* Formato del paquete → fijo y legacy

El packet builder actúa como capa de traducción entre ambos.

---

## Mejoras futuras

Posibles mejoras:

* Definir un nuevo formato de paquete (v2)
* Añadir versionado
* Incluir checksum o integridad
* Implementar parser (operación inversa)


# CookiePacketData

## Objetivo

La estructura `CookiePacketData` representa todos los datos a transmitir de forma clara y estructurada.

Actúa como modelo interno de datos antes de ser convertido en un paquete binario.

El objetivo es desacoplar la representación de datos del formateo del paquete.

---

## Qué hace

`CookiePacketData`:

* Almacena todos los datos relevantes para transmisión
* Agrupa los campos por significado (GNSS, navegación, IMU, estado)
* Sirve de interfaz clara entre los módulos y el packet builder

---

## Estructura

```
CookiePacketData
```

La estructura se compone de bloques agrupados lógicamente:

* Datos GNSS (valores crudos)
* Datos de navegación (valores procesados)
* Datos IMU
* Estado / flags

---

## Datos GNSS

### Objetivo

Representar los datos de posicionamiento crudos obtenidos del GNSS.

### Características

* Salida directa del procesamiento GNSS
* Sin interpretación adicional
* Incluye latitud, longitud, altitud y modo

---

## Datos de navegación

### Objetivo

Representar los datos procesados utilizados por el sistema de navegación.

### Características

* Derivados de la fusión GNSS + IMU
* Expresados en unidades físicas (grados, metros)
* Salida principal del módulo de navegación

---

## Datos IMU

### Objetivo

Representar los datos inerciales incluidos en el paquete.

### Características

* Incluyen aceleración y velocidad angular
* Ya preprocesados y alineados si es necesario
* Sin transformaciones adicionales dentro de la estructura

---

## Estado / flags

### Objetivo

Almacenar información del estado del sistema relacionada con el paquete.

### Ejemplos

* Flag de validez ('A' / 'V')
* Modo GNSS
* Otros indicadores del sistema

---

## Principios de diseño

La estructura sigue:

* Agrupación lógica de datos
* Separación clara de responsabilidades
* Independencia del layout del paquete
* Simplicidad y legibilidad
* Facilidad de testeo

---

## Qué NO hace

`CookiePacketData` no:

* Formatea paquetes
* Maneja bytes u offsets
* Conoce el tamaño del paquete
* Convierte unidades
* Lee sensores
* Ejecuta lógica de navegación o EKF

---

## Rol en el sistema

`CookiePacketData` se sitúa entre:

```
Sensores / Navigation → CookiePacketData → Packet Builder → Paquete binario
```

Actúa como contenedor unificado de datos.

---

## Flujo de ejemplo

```
Datos GNSS
        ↓
Actualización de navegación
        ↓
Procesado IMU
        ↓
CookiePacketData
        ↓
Packet Builder
        ↓
Paquete de 75 bytes
```

---

## Razonamiento de diseño

La implementación original mezclaba:

* Generación de datos
* Formateo del paquete
* Manipulación a nivel de byte

Esta estructura separa el modelo de datos del formato para:

* Mejorar claridad
* Permitir diseño modular
* Facilitar depuración
* Evitar código acoplado

---

## Comportamiento esperado

* Los datos se almacenan de forma clara
* Los campos son accesibles y modificables fácilmente
* Puede usarse independientemente del packet builder

---

## Mejoras futuras

Posibles extensiones:

* Añadir nuevos campos sin afectar el layout
* Soportar múltiples formatos de paquete
* Introducir versionado
* Ampliar información de estado
