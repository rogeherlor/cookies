# Inartrans_porting Project Goal

The goal of this project is to port and restructure the cookie firmware so that it works with a current version of Simplicity Studio.

The existing code will be used as a functional reference, but not as a direct base. The intention is to rewrite the system in a modular, clear, and maintainable way.

## Expected Functionality

The cookie must be able to:

- Initialize the required hardware.
- Read IMU data.
- Read GNSS data via UART.
- Parse GNSS NMEA sentences.
- Use the existing EKF to combine IMU and GNSS data.
- Build data packets.
- Send those packets over the radio.
- Receive and forward messages when acting as an intermediate node.
- Operate either as a sensor node or as a coordinator depending on configuration.

## Design Criteria

The new code should clearly separate responsibilities:

- Sensors.
- GNSS.
- Navigation/EKF.
- Packet building.
- Network.
- Main application.
- Configuration commands.

The goal is not to change the system behaviour, but to preserve the existing functionality with a cleaner and more structured design.

____________________________________________________
____________________________________________________


# Objetivo del proyecto Inartrans_porting

El objetivo de este proyecto es portar y reestructurar el código de las cookies para que funcione en una versión actual de Simplicity Studio.

El código antiguo se usará como referencia funcional, pero no como base directa. La intención es reescribir el sistema de forma modular, clara y mantenible.

## Funcionalidad esperada

La cookie debe ser capaz de:

- Inicializar el hardware necesario.
- Leer datos de la IMU.
- Leer datos GNSS por UART.
- Parsear las tramas NMEA del GNSS.
- Usar el EKF existente para combinar IMU y GNSS.
- Construir paquetes de datos.
- Enviar esos paquetes por radio.
- Recibir y reenviar mensajes si actúa como nodo intermedio.
- Actuar como nodo sensor o coordinador según la configuración.

## Criterio de diseño

El nuevo código debe separar responsabilidades:

- Sensores.
- GNSS.
- Navegación/EKF.
- Construcción de paquetes.
- Red.
- Aplicación principal.
- Comandos de configuración.

El objetivo no es cambiar el comportamiento del sistema, sino conservar la funcionalidad existente con una estructura más limpia.