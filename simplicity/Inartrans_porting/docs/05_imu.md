# IMU Module

## Purpose

The IMU module provides a clean and hardware-independent representation of inertial measurements.

It separates raw sensor data, unit conversion, and navigation-axis preprocessing so that IMU data can be safely integrated into the navigation system.

## Structure

The IMU pipeline is divided into three layers:

    IMU sample → IMU converter → IMU preprocessor → Navigation

## IMU Sample

### Purpose

Represents raw IMU data exactly as it is received from the sensor or test input.

### Structure

    CookieImuSample

Contains:

- Acceleration in milli-g (`accel_mg`)
- Angular velocity in degrees per second (`gyro_dps`)
- Timestamp in milliseconds (`timestamp_ms`)
- Validity flag

### Characteristics

- No unit conversion
- No physics applied
- No axis transformation
- Direct mapping from hardware or test data
- Fully testable without hardware

## IMU Converter

### Purpose

Transforms raw IMU data into physical units required by the EKF.

### Input

    CookieImuSample

### Output

    CookieImuConvertedSample

Contains:

- Acceleration in m/s²
- Angular velocity in rad/s
- Time step `dt` in seconds
- Validity flag

### What it does

1. Converts acceleration:

       mg → m/s²

2. Converts angular velocity:

       deg/s → rad/s

3. Computes time difference:

       dt = current_timestamp - previous_timestamp

### What it does NOT do

- Sensor reading
- Filtering
- Bias correction
- Axis alignment
- Gravity compensation
- EKF logic

## IMU Preprocessor

### Purpose

Adapts converted IMU data to the axis convention expected by the navigation/EKF layer.

This step is based on the original `Inartrans_v2` implementation, which applied an axis transformation before calling `EKF_Predict()`.

### Input

    CookieImuConvertedSample

### Output

    CookieImuNavigationInput

Contains:

- Acceleration in m/s², aligned for navigation
- Angular velocity in rad/s, aligned for navigation
- Time step `dt` in seconds
- Validity flag

### Axis mapping

The original implementation applied the following mapping before EKF prediction:

    x_nav = -x_sensor
    y_nav = -z_sensor
    z_nav = -y_sensor

The same mapping is applied to both acceleration and gyroscope data.

### What it does NOT do

- Unit conversion
- Sensor reading
- Filtering
- Bias correction
- Gravity compensation
- EKF prediction

## Design rationale

The original system mixed:

- Sensor reading
- Unit conversion
- Axis alignment
- Navigation logic

This module separates responsibilities to:

- Improve clarity
- Enable unit testing
- Avoid hidden dependencies
- Preserve the working axis convention from the old system
- Prepare clean input for `EKF_Predict()`

## Example flow

    Raw IMU data
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    Physical IMU data (m/s², rad/s, dt)
            ↓
    IMU preprocessor
            ↓
    Navigation-aligned IMU data
            ↓
    Navigation module / EKF_Predict()

____________________________________________________
____________________________________________________

# Módulo IMU

## Objetivo

El módulo IMU proporciona una representación limpia e independiente del hardware de las medidas inerciales.

Separa los datos crudos del sensor, la conversión de unidades y el preprocesado de ejes para que los datos IMU puedan integrarse de forma segura en el sistema de navegación.

## Estructura

El flujo del IMU se divide en tres capas:

    IMU sample → IMU converter → IMU preprocessor → Navigation

## IMU Sample

### Objetivo

Representa los datos crudos del IMU tal como los entrega el sensor o una entrada de test.

### Estructura

    CookieImuSample

Contiene:

- Aceleración en milli-g (`accel_mg`)
- Velocidad angular en grados por segundo (`gyro_dps`)
- Timestamp en milisegundos (`timestamp_ms`)
- Flag de validez

### Características

- Sin conversión de unidades
- Sin física aplicada
- Sin transformación de ejes
- Mapeo directo del hardware o tests
- Totalmente testeable sin sensor

## IMU Converter

### Objetivo

Convierte los datos crudos en unidades físicas necesarias para el EKF.

### Entrada

    CookieImuSample

### Salida

    CookieImuConvertedSample

Contiene:

- Aceleración en m/s²
- Velocidad angular en rad/s
- Tiempo `dt` en segundos
- Flag de validez

### Qué hace

1. Conversión de aceleración:

       mg → m/s²

2. Conversión de velocidad angular:

       deg/s → rad/s

3. Cálculo de delta temporal:

       dt = diferencia entre timestamps

### Qué NO hace

- Lectura del sensor
- Filtrado
- Corrección de bias
- Alineación de ejes
- Compensación de gravedad
- Lógica del EKF

## IMU Preprocessor

### Objetivo

Adapta los datos IMU ya convertidos a la convención de ejes esperada por la capa de navegación/EKF.

Este paso se basa en la implementación original de `Inartrans_v2`, donde se aplicaba una transformación de ejes antes de llamar a `EKF_Predict()`.

### Entrada

    CookieImuConvertedSample

### Salida

    CookieImuNavigationInput

Contiene:

- Aceleración en m/s² alineada para navegación
- Velocidad angular en rad/s alineada para navegación
- Tiempo `dt` en segundos
- Flag de validez

### Mapeo de ejes

La implementación original aplicaba el siguiente mapeo antes de la predicción del EKF:

    x_nav = -x_sensor
    y_nav = -z_sensor
    z_nav = -y_sensor

El mismo mapeo se aplica a aceleración y giroscopio.

### Qué NO hace

- Conversión de unidades
- Lectura del sensor
- Filtrado
- Corrección de bias
- Compensación de gravedad
- Predicción del EKF

## Razonamiento de diseño

El sistema original mezclaba:

- Lectura de sensores
- Conversión de unidades
- Alineación de ejes
- Lógica de navegación

Este módulo separa responsabilidades para:

- Mejorar claridad
- Permitir testing
- Evitar dependencias ocultas
- Preservar la convención de ejes que funcionaba en el sistema antiguo
- Preparar datos limpios para `EKF_Predict()`

## Flujo de ejemplo

    Datos IMU crudos
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    Datos físicos (m/s², rad/s, dt)
            ↓
    IMU preprocessor
            ↓
    Datos IMU alineados para navegación
            ↓
    Módulo Navigation / EKF_Predict()

