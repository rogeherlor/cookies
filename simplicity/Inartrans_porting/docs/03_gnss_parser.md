# GNSS Parser and Converter Modules

## Purpose

The GNSS modules convert raw NMEA text data into structured and usable information for the rest of the system.

They do **not** interact with hardware, UART, or sensors directly. Their responsibility is to interpret already received GNSS data and prepare it for later modules, such as navigation or packet building.

## Module split

The GNSS functionality is currently divided into two small modules:

- `gnss.c / gnss.h` → parses NMEA text into a `CookieGnssFix` structure.
- `gnss_converter.c / gnss_converter.h` → converts raw NMEA coordinates into decimal degrees.

This separation keeps string parsing and coordinate conversion independent.

## What the parser does

The parser takes a buffer containing NMEA sentences, usually one GNSS epoch, and performs the following steps:

1. **Sentence detection**
   - Searches for NMEA sentences starting with `$`.

2. **Checksum validation**
   - Verifies that each sentence is not corrupted using the NMEA checksum.

3. **Sentence parsing**
   - Extracts relevant information from:
     - `RMC` → time, validity, speed, course, date
     - `GGA` → latitude, longitude, altitude
     - `GSA` → PDOP, which gives information about positioning quality

4. **Data conversion**
   - Converts text fields into basic numeric values when needed.

5. **Data aggregation**
   - Combines all relevant information into a single structure: `CookieGnssFix`.

6. **Epoch validation**
   - Only returns valid parsed data if all required sentences are present.

## Parser input

The parser receives:

    const char *buffer;
    uint32_t length;

Where:

- `buffer` contains multiple NMEA sentences.
- `length` is the number of characters available in the buffer.

## Parser output

The parser fills:

    CookieGnssFix fix;

This structure contains:

- UTC time
- Date
- Latitude and longitude in raw NMEA format
- Altitude
- Speed
- Course over ground
- PDOP
- Validity flag

## What the converter does

The converter receives a parsed `CookieGnssFix` and converts latitude and longitude from raw NMEA format to decimal degrees.

Example:

    4807.038 N  →  48.117302
    01131.000 E →  11.516666

The conversion follows:

    decimal degrees = degrees + minutes / 60

The direction field is also applied:

- `N` and `E` → positive values
- `S` and `W` → negative values

## Converter input

The converter receives:

    const CookieGnssFix *fix;
    float *latitude_deg;
    float *longitude_deg;

## Converter output

The converter writes:

    latitude_deg
    longitude_deg

These values are suitable for modules that expect geodetic latitude and longitude in decimal degrees, such as the EKF navigation layer.

## What these modules do NOT do

These modules intentionally avoid:

- Hardware access
- UART handling
- Interrupt handling
- GNSS configuration or mode selection
- Global variables
- EKF or navigation logic
- Packet building
- Network communication

## Design rationale

The original system mixed several responsibilities:

- UART handling
- Buffer management
- NMEA parsing
- Global variable updates
- GNSS mode logic
- EKF-related logic

The new design separates these responsibilities:

- The parser only understands NMEA text.
- The converter only transforms coordinates.
- Other modules decide how to use the parsed and converted data.

This improves readability, enables testing without hardware, avoids hidden dependencies, and makes the system easier to maintain and port.

## Example flow

    Raw NMEA buffer
            ↓
    GNSS parser
            ↓
    CookieGnssFix structure
            ↓
    GNSS converter
            ↓
    Decimal latitude / longitude
            ↓
    Application / navigation / packet modules

____________________________________________________
____________________________________________________

# Módulos GNSS Parser y Converter

## Objetivo

Los módulos GNSS convierten datos de texto NMEA en información estructurada y utilizable por el resto del sistema.

No interactúan directamente con el hardware, la UART ni los sensores. Su responsabilidad es interpretar datos GNSS ya recibidos y prepararlos para módulos posteriores, como navegación o construcción de paquetes.

## División de módulos

La funcionalidad GNSS está dividida actualmente en dos módulos pequeños:

- `gnss.c / gnss.h` → parsea texto NMEA y lo convierte en una estructura `CookieGnssFix`.
- `gnss_converter.c / gnss_converter.h` → convierte coordenadas NMEA brutas a grados decimales.

Esta separación mantiene independiente el parseo de texto y la conversión de coordenadas.

## Qué hace el parser

El parser recibe un buffer con sentencias NMEA, normalmente una época GNSS, y realiza los siguientes pasos:

1. **Detección de sentencias**
   - Busca sentencias NMEA que empiezan por `$`.

2. **Validación de checksum**
   - Comprueba que cada sentencia no está corrupta usando el checksum NMEA.

3. **Parseo de sentencias**
   - Extrae información relevante de:
     - `RMC` → tiempo, validez, velocidad, rumbo, fecha
     - `GGA` → latitud, longitud, altitud
     - `GSA` → PDOP, que da información sobre la calidad del posicionamiento

4. **Conversión de datos**
   - Convierte campos de texto en valores numéricos básicos cuando hace falta.

5. **Agregación**
   - Junta toda la información relevante en una única estructura: `CookieGnssFix`.

6. **Validación de época**
   - Solo devuelve datos parseados válidos si están presentes todas las sentencias necesarias.

## Entrada del parser

El parser recibe:

    const char *buffer;
    uint32_t length;

Donde:

- `buffer` contiene varias sentencias NMEA.
- `length` es el número de caracteres disponibles en el buffer.

## Salida del parser

El parser rellena:

    CookieGnssFix fix;

Esta estructura contiene:

- Tiempo UTC
- Fecha
- Latitud y longitud en formato NMEA bruto
- Altitud
- Velocidad
- Rumbo
- PDOP
- Flag de validez

## Qué hace el converter

El converter recibe una estructura `CookieGnssFix` ya parseada y convierte la latitud y longitud desde formato NMEA bruto a grados decimales.

Ejemplo:

    4807.038 N  →  48.117302
    01131.000 E →  11.516666

La conversión sigue:

    grados decimales = grados + minutos / 60

También se aplica el campo de dirección:

- `N` y `E` → valores positivos
- `S` y `W` → valores negativos

## Entrada del converter

El converter recibe:

    const CookieGnssFix *fix;
    float *latitude_deg;
    float *longitude_deg;

## Salida del converter

El converter escribe:

    latitude_deg
    longitude_deg

Estos valores son adecuados para módulos que esperan latitud y longitud geodésicas en grados decimales, como la capa de navegación/EKF.

## Qué NO hacen estos módulos

Estos módulos evitan intencionadamente:

- Acceso al hardware
- Gestión de UART
- Gestión de interrupciones
- Configuración o selección de modo GNSS
- Variables globales
- Lógica EKF o de navegación
- Construcción de paquetes
- Comunicación por red

## Razonamiento de diseño

El sistema original mezclaba varias responsabilidades:

- Gestión de UART
- Gestión de buffers
- Parseo NMEA
- Actualización de variables globales
- Lógica de modos GNSS
- Lógica relacionada con el EKF

El nuevo diseño separa estas responsabilidades:

- El parser solo interpreta texto NMEA.
- El converter solo transforma coordenadas.
- Otros módulos deciden cómo usar los datos parseados y convertidos.

Esto mejora la legibilidad, permite pruebas sin hardware, evita dependencias ocultas y facilita el mantenimiento y el portado del sistema.

## Flujo de ejemplo

    Buffer NMEA bruto
            ↓
    Parser GNSS
            ↓
    Estructura CookieGnssFix
            ↓
    Converter GNSS
            ↓
    Latitud / longitud decimal
            ↓
    Aplicación / navegación / paquetes
