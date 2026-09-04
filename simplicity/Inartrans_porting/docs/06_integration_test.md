# Integration Test

## Purpose

The integration test checks that the main software modules can work together in a complete data flow.

It uses simulated GNSS and IMU inputs and a mock EKF. The goal is to validate module integration, not the physical correctness of the real EKF.

## What it tests

The test connects:

- GNSS parser/converter
- IMU sample
- IMU converter
- IMU preprocessor
- Navigation module
- Mock EKF

## Test flow

    Fake GNSS data
            ↓
    Navigation update
            ↓
    EKF_Init / EKF_Update through mock EKF

    Fake IMU data
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    IMU preprocessor
            ↓
    Navigation prediction
            ↓
    EKF_Predict through mock EKF

## Expected behaviour

- The navigation module waits for several valid GNSS fixes before initialising.
- Once initialised, the navigation state becomes available.
- With zero acceleration and zero angular velocity, the mock velocity remains zero.
- The latitude, longitude and altitude remain stable.

## What it does NOT validate

This test does not validate:

- Real EKF behaviour
- Physical navigation accuracy
- CMSIS/DSP integration
- Simplicity Studio integration
- Hardware drivers
- UART input
- Real IMU data
- Real GNSS data
- Radio communication

## Why a mock EKF is used

The real EKF depends on CMSIS/DSP and the Simplicity build environment.

Using a mock EKF allows the architecture and data flow to be tested locally without depending on hardware, Simplicity Studio, or the EKF implementation maintained externally.

____________________________________________________
____________________________________________________

# Test de integración

## Objetivo

El test de integración comprueba que los módulos principales pueden trabajar juntos en un flujo completo de datos.

Usa entradas GNSS e IMU simuladas y un EKF mock. El objetivo es validar la integración entre módulos, no la corrección física del EKF real.

## Qué prueba

El test conecta:

- Parser/converter GNSS
- IMU sample
- IMU converter
- IMU preprocessor
- Módulo Navigation
- EKF mock

## Flujo del test

    Datos GNSS simulados
            ↓
    Actualización de navegación
            ↓
    EKF_Init / EKF_Update mediante EKF mock

    Datos IMU simulados
            ↓
    IMU sample
            ↓
    IMU converter
            ↓
    IMU preprocessor
            ↓
    Predicción de navegación
            ↓
    EKF_Predict mediante EKF mock

## Comportamiento esperado

- El módulo de navegación espera varios fixes GNSS válidos antes de inicializarse.
- Una vez inicializado, el estado de navegación está disponible.
- Con aceleración y velocidad angular nulas, la velocidad mock se mantiene a cero.
- La latitud, longitud y altitud permanecen estables.

## Qué NO valida

Este test no valida:

- Comportamiento del EKF real
- Precisión física de navegación
- Integración con CMSIS/DSP
- Integración con Simplicity Studio
- Drivers de hardware
- Entrada UART
- Datos IMU reales
- Datos GNSS reales
- Comunicación por radio

## Por qué se usa un EKF mock

El EKF real depende de CMSIS/DSP y del entorno de compilación de Simplicity.

Usar un EKF mock permite probar localmente la arquitectura y el flujo de datos sin depender del hardware, de Simplicity Studio ni de la implementación del EKF mantenida externamente.

