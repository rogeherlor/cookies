# GNSS Parser Module

## Purpose

The GNSS parser module converts raw NMEA text data into a structured and usable format for the rest of the system.

It does **not** interact with hardware, UART, or sensors directly. Its only responsibility is to interpret already received GNSS data.

## What it does

The module takes a buffer containing NMEA sentences, usually one GNSS epoch, and performs the following steps:

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
   - Converts text fields into numeric values when needed.

5. **Data aggregation**
   - Combines all relevant information into a single structure: `CookieGnssFix`.

6. **Epoch validation**
   - Only returns valid parsed data if all required sentences are present.

## Input

The parser receives:

    const char *buffer;
    uint32_t length;

Where:

- `buffer` contains multiple NMEA sentences.
- `length` is the number of characters available in the buffer.

## Output

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

## What it does NOT do

This module intentionally avoids:

- Hardware access
- UART handling
- Interrupt handling
- GNSS configuration or mode selection
- Coordinate conversion to decimal degrees
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

This module isolates only the parsing logic to:

- Improve readability
- Enable testing without hardware
- Avoid hidden dependencies
- Make the system easier to maintain and port

## Example flow

    Raw NMEA buffer
            ↓
    GNSS parser
            ↓
    CookieGnssFix structure
            ↓
    Application / navigation / packet modules

____________________________________________________
____________________________________________________

# Módulo GNSS Parser

## Objetivo

El módulo GNSS parser convierte datos de texto NMEA en una estructura clara y utilizable por el resto del sistema.

No interactúa directamente con el hardware, la UART ni los sensores. Su única responsabilidad es interpretar datos GNSS ya recibidos.

## Qué hace

El módulo recibe un buffer con sentencias NMEA, normalmente una época GNSS, y realiza los siguientes pasos:

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
   - Convierte campos de texto en valores numéricos cuando hace falta.

5. **Agregación**
   - Junta toda la información relevante en una única estructura: `CookieGnssFix`.

6. **Validación de época**
   - Solo devuelve datos parseados válidos si están presentes todas las sentencias necesarias.

## Entrada

El parser recibe:

    const char *buffer;
    uint32_t length;

Donde:

- `buffer` contiene varias sentencias NMEA.
- `length` es el número de caracteres disponibles en el buffer.

## Salida

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

## Qué NO hace

Este módulo evita intencionadamente:

- Acceso al hardware
- Gestión de UART
- Gestión de interrupciones
- Configuración o selección de modo GNSS
- Conversión de coordenadas a grados decimales
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

Este módulo aísla únicamente la lógica de parseo para:

- Mejorar la legibilidad
- Permitir pruebas sin hardware
- Evitar dependencias ocultas
- Facilitar el mantenimiento y el portado del sistema

## Flujo de ejemplo

    Buffer NMEA bruto
            ↓
    Parser GNSS
            ↓
    Estructura CookieGnssFix
            ↓
    Aplicación / navegación / paquetes
