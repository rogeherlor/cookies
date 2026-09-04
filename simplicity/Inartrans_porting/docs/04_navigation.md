# Navigation Module

## Purpose

The navigation module connects GNSS and IMU data with the existing EKF to produce a consistent and usable navigation state.

It does **not** implement the EKF itself. Instead, it wraps the existing EKF interface and feeds it with properly formatted data.

## What it does

The module performs the following steps:

1. **Receives GNSS fixes**
   - Input comes from `CookieGnssFix` (GNSS parser output).

2. **Converts GNSS coordinates**
   - Uses the GNSS converter to obtain latitude and longitude in decimal degrees.

3. **Initialises the EKF**
   - Waits for several consecutive valid GNSS fixes before calling:
     - `EKF_Init(lat0, lon0, alt0)`
   - This avoids using unstable initial measurements.

4. **Updates the EKF with GNSS**
   - For each valid GNSS fix after initialisation:
     - `EKF_Update(lat, lon, alt)`

5. **Receives preprocessed IMU input**
   - Input comes from `CookieImuNavigationInput`.
   - This data has already been converted to physical units and aligned to the navigation axis convention.

6. **Runs EKF prediction with IMU**
   - For each valid IMU input:
     - `EKF_Predict(accel_m_s2, gyro_rad_s, dt_s)`

7. **Retrieves navigation state**
   - Converts internal ENU state back to LLA:
     - `ENU_to_LLA(...)`
   - Computes velocity magnitude.

8. **Provides a clean output**
   - Returns a `CookieNavigationState` structure.

## Input

GNSS input:

    CookieGnssFix fix;

Requirements:

- `fix.valid == true`
- Valid latitude/longitude after conversion

IMU input:

    CookieImuNavigationInput imu;

Requirements:

- `imu.valid == true`
- `imu.dt_s > 0`
- Acceleration in m/s²
- Angular velocity in rad/s
- Axes already aligned for navigation

## Output

    CookieNavigationState state;

Contains:

- Latitude (decimal degrees)
- Longitude (decimal degrees)
- Altitude (meters)
- Velocity (m/s)
- Validity flag

## What it does NOT do

This module intentionally avoids:

- EKF implementation
- Sensor reading
- UART handling
- Raw IMU unit conversion
- IMU axis preprocessing
- Packet building
- Network communication

## Design rationale

The original system mixed:

- GNSS parsing
- IMU preprocessing
- EKF logic
- State updates
- Hardware interaction

This module isolates only the **integration layer** between prepared sensor data and the EKF:

- GNSS parser/converter prepares GNSS data.
- IMU converter/preprocessor prepares IMU data.
- Navigation calls the EKF public interface.

This keeps responsibilities clear, avoids modifying the EKF implementation, and makes the system easier to extend and test.

## External dependency

This module depends on:

    cookie_ekf.h

The EKF is maintained externally and must be provided by the main project.

## Example flow

    Raw NMEA buffer
            ↓
    GNSS parser
            ↓
    GNSS converter
            ↓
            ┐
             ─→ Navigation module → EKF_Update()
            ┘

    Raw IMU data
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    IMU preprocessor
            ↓
            ┐
             ─→ Navigation module → EKF_Predict()
            ┘

    EKF state
            ↓
    Navigation state (LLA + velocity)

____________________________________________________
____________________________________________________

# Módulo Navigation

## Objetivo

El módulo de navegación conecta los datos GNSS e IMU con el EKF existente para generar un estado de navegación coherente y utilizable.

No implementa el EKF. Solo lo envuelve y le proporciona los datos en el formato adecuado.

## Qué hace

El módulo realiza los siguientes pasos:

1. **Recibe fixes GNSS**
   - Entrada desde `CookieGnssFix` (salida del parser GNSS).

2. **Convierte coordenadas GNSS**
   - Usa el converter para obtener latitud y longitud en grados decimales.

3. **Inicializa el EKF**
   - Espera varios fixes GNSS válidos consecutivos antes de llamar a:
     - `EKF_Init(lat0, lon0, alt0)`
   - Evita inicializar con datos inestables.

4. **Actualiza el EKF con GNSS**
   - Para cada fix válido después de la inicialización:
     - `EKF_Update(lat, lon, alt)`

5. **Recibe entrada IMU preprocesada**
   - Entrada desde `CookieImuNavigationInput`.
   - Estos datos ya están convertidos a unidades físicas y alineados con la convención de ejes de navegación.

6. **Ejecuta la predicción del EKF con IMU**
   - Para cada entrada IMU válida:
     - `EKF_Predict(accel_m_s2, gyro_rad_s, dt_s)`

7. **Obtiene el estado de navegación**
   - Convierte el estado interno ENU a LLA:
     - `ENU_to_LLA(...)`
   - Calcula la velocidad.

8. **Devuelve un estado limpio**
   - A través de `CookieNavigationState`.

## Entrada

Entrada GNSS:

    CookieGnssFix fix;

Requisitos:

- `fix.valid == true`
- Coordenadas válidas tras conversión

Entrada IMU:

    CookieImuNavigationInput imu;

Requisitos:

- `imu.valid == true`
- `imu.dt_s > 0`
- Aceleración en m/s²
- Velocidad angular en rad/s
- Ejes ya alineados para navegación

## Salida

    CookieNavigationState state;

Contiene:

- Latitud (grados decimales)
- Longitud (grados decimales)
- Altitud (metros)
- Velocidad (m/s)
- Flag de validez

## Qué NO hace

Este módulo evita intencionadamente:

- Implementar el EKF
- Lectura de sensores
- Gestión de UART
- Conversión de unidades IMU crudas
- Preprocesado de ejes IMU
- Construcción de paquetes
- Comunicación de red

## Razonamiento de diseño

El sistema original mezclaba:

- Parseo GNSS
- Preprocesado IMU
- Lógica del EKF
- Actualización de estados
- Interacción con hardware

Este módulo aísla únicamente la **capa de integración** entre datos de sensores ya preparados y el EKF:

- El parser/converter GNSS prepara los datos GNSS.
- El converter/preprocessor IMU prepara los datos IMU.
- Navigation llama a la interfaz pública del EKF.

Esto mantiene responsabilidades claras, evita modificar el EKF y facilita la extensión y el testeo del sistema.

## Dependencia externa

Este módulo depende de:

    cookie_ekf.h

El EKF se mantiene fuera y debe ser proporcionado por el proyecto principal.

## Flujo de ejemplo

    Buffer NMEA bruto
            ↓
    Parser GNSS
            ↓
    Converter GNSS
            ↓
            ┐
             ─→ Módulo Navigation → EKF_Update()
            ┘

    Datos IMU crudos
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    IMU preprocessor
            ↓
            ┐
             ─→ Módulo Navigation → EKF_Predict()
            ┘

    Estado EKF
            ↓
    Estado de navegación (LLA + velocidad)

