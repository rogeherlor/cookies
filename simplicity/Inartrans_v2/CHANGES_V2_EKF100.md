# Inartrans_v2 — Changes: IMU/GNSS data acquisition improvements + EKF at 100 Hz

## Overview

This document records the changes applied to `Inartrans_v2` to bring its IMU and GNSS data
acquisition in line with the fixes developed in `Inartrans_v2_imu`, while additionally running
the Extended Kalman Filter (EKF) prediction step at **100 Hz** instead of the radio report
rate of ~4 Hz.

The 75-byte data packet (`paquete[75]`) byte layout is **unchanged**. All `memcpy(paquete+N, ...)`
offsets are identical to the original `Inartrans_v2`.

---

## Files changed

### 1. `main/EFR32/main.c`

| Before | After |
|--------|-------|
| `IMU_config(20)` | `IMU_config(200)` |

**Reason:** The ICM20648 was configured at 20 Hz but `IMU_getAccelerometerData` /
`IMU_getGyroData` were called at 200 Hz, returning the same stale sample ~90% of the time.
Setting the sensor rate to 200 Hz ensures every call returns a fresh measurement.

---

### 2. `imu.c`

| Before | After |
|--------|-------|
| `ICM20648_accelBandwidthSet(ICM20648_ACCEL_BW_6HZ)` | `ICM20648_accelBandwidthSet(ICM20648_ACCEL_BW_24HZ)` |
| `ICM20648_gyroBandwidthSet(ICM20648_GYRO_BW_6HZ)` | `ICM20648_gyroBandwidthSet(ICM20648_GYRO_BW_24HZ)` |

**Reason:** A 6 Hz bandwidth filter introduces ~83 ms of phase lag and attenuates all
signal content above ~4 Hz, killing navigation dynamics. A 24 Hz bandwidth reduces phase
lag to ~21 ms and preserves the 0–10 Hz range useful for pedestrian/vehicle navigation.

---

### 3. `hal/com.c` — `USART1_RX_IRQHandler`

Added `extern volatile uint8_t gnss_lines_received;` declaration after the existing
includes.

Replaced the old ISR body with:

```c
void USART1_RX_IRQHandler(void)
{
  uint32_t flags = USART_IntGet(USART1);
  USART_IntClear(USART1, flags);
  uint8_t c = USART_RxDataGet(USART1);

  if (indice < MI_BUFFER_SIZE - 1) {
    mi_buffer[indice] = c;
    indice++;
    tengo_gps = 1;
    if (c == '\n') { gnss_lines_received++; }
  } else {
    // Buffer overflow: reset and start fresh
    for (int i = 0; i < MI_BUFFER_SIZE; i++) mi_buffer[i] = '0';
    indice = 0;
    gnss_lines_received = 0;
  }
}
```

**Reason (old ISR):** The old ISR called the character input handler which parsed NMEA
inline inside an ISR — unsafe and slow. It also had no overflow protection.

**Reason (new ISR):** Stores raw bytes in `mi_buffer`, increments `gnss_lines_received` on
each `'\n'`, and performs an atomic overflow reset if the buffer fills up. All parsing is
deferred to `emberAfMainTickCallback`.

---

### 4. `gps-uart.c` and `gps-uart.h`

Replaced with the `Inartrans_v2_imu` versions.

**Key difference:** The old `busca2()` multi-pass NMEA parser was replaced by
`parse_nmea_epoch()` — a single O(n) forward pass with:
- NMEA `*XX` checksum validation on every sentence
- Order-independent detection of RMC, GGA, and GSA sentences
- Single commit of all parsed fields when a complete epoch (all three sentences) is found

**Variables added/exported:** `fecha[]`, `PDOP[]`, `vel_GNSS[]`, `cog_GNSS[]`.

---

### 5. `flex-callbacks.c` — major restructure

#### Architecture before (v2 original)

`reportHandler()` did everything at ~4 Hz (250 ms period):
- Read IMU
- Call `busca2()` to parse GNSS
- Run `EKF_Predict` with a variable `Ts1` derived from RTC difference
- Assemble and transmit the 75-byte packet

#### Architecture after

**`emberAfMainTickCallback()`** handles all sensor work:
- IMU at 200 Hz via RTCDRV periodic timer
- EKF predict at 100 Hz (every 2nd IMU sample, fixed Ts = 0.01 s)
- GNSS parsing event-driven on `gnss_lines_received >= 6`

**`reportHandler()`** is simplified to:
- Advance GNSS mode selector counter
- Compute EKF output (`ENU_to_LLA`, velocity magnitude) — **unconditionally** (see below)
- Build `tiempo_out` from `gps_ms_ref + elapsed RTC` — **unconditionally**
- Assemble 75-byte packet from cached state — only when connected
- Transmit via `manda_datos()` — only when connected

#### New module-scope variables added

```c
// IMU 200 Hz timer
static RTCDRV_TimerID_t imu_timer_id;
static volatile bool imu_sample_pending = false;

// GNSS epoch detection
volatile uint8_t gnss_lines_received = 0;
#define GNSS_LINES_PER_EPOCH  6

// GPS-synced timestamp
static uint32_t gps_ms_ref     = 0;
static uint32_t rtc_at_gps_fix = 0;
static bool     gps_time_valid = false;

// Cached sensor state for packet assembly
static int32_t last_acelint[3] = {0, 0, 0};
static int32_t last_gyroint[3] = {0, 0, 0};
static float   latitud_s       = 0.0f;
static float   longitud_s      = 0.0f;
static float   altitud_s       = 0.0f;
```

#### Serial log format

All serial prints use integer-scaled values since `emberAfCorePrint` does not support `%f`.

| Print prefix | Source | Condition | Fields and scaling |
|---|---|---|---|
| `IMU t=...; A=...; G=...` | `emberAfMainTickCallback` | Every 5 ms (200 Hz) | `t` = ms since UTC midnight (GPS-synced or raw RTC); `A` = accel × 1000 (g); `G` = gyro × 100 (deg/s, 2 dp) |
| `GP V=A; Mod=...; T=...; D=...; lat=...; lon=...; alt=...; vel=...; COG=...; PDOP=...` or same with `GN` | `emberAfMainTickCallback` | Every GNSS epoch (~1 Hz), valid fix | `Mod` = active `GNSS_mode` (1–7); `T` = ms since UTC midnight; `D` = date as DDMMYY integer; `lat`/`lon` × 10000 (NMEA DDMM.MMMM × 10000); `alt` × 100 (cm); `vel` = cm/s; `COG` × 100 (rad × 100); `PDOP` × 100 |
| `EKF lat=...; lon=...; alt=...; vel=...` | `reportHandler` | Every 250 ms, once EKF initialised | `lat`/`lon` × 10000 (decimal degrees × 10000); `alt` × 100 (cm); `vel` × 100 (cm/s) |

**GNSS talker detection:** On each epoch, `mi_buffer` is scanned for evidence of
multi-constellation operation. `gnss_talker` is set to `"GN"` if **any** of the following
are found in the buffer:

- `$GNRMC` or `$GNGGA` — navigation sentences with the GN talker
- `$GLGSA` or `$GLGSV` — GLONASS DOP / satellite-in-view sentences
- `$GAGSA` or `$GAGSV` — Galileo equivalents

If none are found, `gnss_talker` is set to `"GP"` (GPS-only module or single-constellation mode).

The result locks after 3 consecutive agreeing epochs and re-detects automatically whenever
`GNSS_mode` changes.

**Why not rely solely on the GGA/RMC talker prefix:** Some L86 firmware versions output
`$GPGGA` / `$GPRMC` (GP-prefixed) even in multi-constellation mode. Scanning those sentences
alone would always yield `GP` on such firmware. The presence of `$GLGSA`/`$GAGSA`/`$GLGSV`/
`$GAGSV` is unambiguous — these sentences only appear when GLONASS or Galileo satellites are
being tracked.

Combined with `Mod=`, the print allows hardware module identification at a glance:

| Print prefix | `Mod` value | Interpretation |
|---|---|---|
| `GP` | any | Quectel **L80** (GPS-only hardware) — always `GP` regardless of mode commands |
| `GP` | 1, 2, 3, 4 | Quectel **L86** in a single-constellation mode — GPS-only sentences expected |
| `GN` | 5, 6, 7 | Quectel **L86** in multi-constellation mode — confirmed GNSS reception |

#### Variables removed from module scope

- `tiempo_referencia`, `tiempo_actual`, `tiempo_anterior`, `tiempo_incremento`, `Ts1_sum`,
  `t_max_ciclo`, `flag_gps0` — replaced by `gps_ms_ref` / `rtc_at_gps_fix`
- `extern uint8_t tiempo_ultimo_valido[10]` — no longer needed
- `tengo_datos_gps` changed from `uint8_t` to `static bool`

---

## IMU subsampling: 200 Hz acquisition → 100 Hz EKF predict

The IMU RTCDRV periodic timer fires every **5 ms (≈200 Hz)**. Inside
`emberAfMainTickCallback()`, a static counter `ekf_div` is incremented on every IMU sample
and reset to 0 when it reaches 2. `EKF_Predict()` is called only on the reset — i.e.,
every **2nd sample = every ~10 ms ≈ 100 Hz**.

### Note on actual sample regularity

RTCDRV uses the EFR32 RTCC peripheral which ticks at **4096 Hz** (one tick = 0.244 ms). The
requested 5 ms period rounds to the nearest tick, so each ISR firing can be **±0.244 ms**
away from the nominal 5 ms interval. This means the IMU does **not** sample at a perfectly
constant 200 Hz — individual intervals carry a small jitter of up to ±0.244 ms.

### Fixed Ts passed to EKF_Predict

The time step passed to `EKF_Predict` is the **fixed constant `Ts = 0.01 s` (10 ms)**, not
a measured RTC interval between successive calls. This is required for two reasons:

1. **Jitter propagation:** Measuring the actual RTC interval between ISR firings would
   directly propagate the ±0.244 ms per-tick error into the filter state estimate, degrading
   dead-reckoning accuracy without any benefit.

2. **Process covariance consistency:** `EKF_Predict` scales the process noise matrix **Q**
   by `Ts` at each step (`P = F·P·Fᵀ + Q·Ts`). If `Ts` were variable, **Q** would be
   effectively different every step, breaking the assumption that **Q** is a tuned constant.
   A filter whose effective **Q** fluctuates is no longer optimally tuned and becomes
   inconsistent (overconfident or underconfident depending on the sign of the jitter).
   The fixed nominal `Ts = 0.01 s` keeps **Q** well-defined and the filter correctly tuned.

### Previous behaviour (original v2)

`EKF_Predict` was called inside `reportHandler()` at **~4 Hz** (250 ms period) using a
variable `Ts1` derived from the RTC difference between consecutive `reportHandler`
invocations. This provided only ~4 prediction steps per second, severely limiting
dead-reckoning accuracy during GNSS outages.

---

## EKF output and timestamp are independent of network connection

In `reportHandler()`, the EKF output computation (`ENU_to_LLA`, velocity magnitude) and the
`tiempo_out` / `fecha_out` timestamp build run **unconditionally for `NODO_QUE_ENVIA = 1`**,
regardless of whether the node has a parent or is connected to the network.

Only the **packet assembly and `manda_datos()` call** remain gated by
`mi_panID != 0xFFFF && mi_rango < 0xFFFF`.

This means:
- The `EKF lat=...` serial print appears as soon as the EKF initialises, even before a parent
  node is found.
- `tiempo_out` is kept up to date continuously, so when the node does connect, the first
  transmitted packet already carries an accurate UTC timestamp.
- `NODO_QUE_ENVIA = 0` (coordinator): `reportHandler` returns immediately at entry — none of
  this runs.

---

## Transmitted frame layout

Every data transmission is 90 bytes: a 15-byte header (`cabeza_trama`) followed by the
75-byte payload (`paquete`). Both are assembled in little-endian byte order (EFR32 is
Cortex-M4, little-endian).

---

### Header — `cabeza_trama` (bytes 0–14 of the air frame)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 1 | `uint8_t` | `tipo` | Message type. `PQT_DATOS` for sensor data packets. |
| 1 | 2 | `uint16_t` | `mi_rango` | Hop distance of the sending node from the coordinator (used for routing). `0xFFFF` = not connected. |
| 3 | 2 | `uint16_t` | `destino` | Final destination node ID. Always `0x0000` (coordinator) for data packets. |
| 5 | 2 | `uint16_t` | `mi_panID` | PAN ID of the network. |
| 7 | 2 | `uint16_t` | `origen` | Original source node ID (2 LSB bytes of the EUI-64 MAC). Never modified by relay nodes. |
| 9 | 2 | `uint16_t` | `numpaq` | Packet counter of the originating node. Counts from 1 to 0xFFFE then wraps to 1. |
| 11 | 2 | `uint16_t` | `rangorigen` | Hop distance of the original source node. |
| 13 | 2 | `uint16_t` | `mi_numseq` | Request/sequence counter of the originating node. |

---

### Payload — `paquete[75]` (bytes 15–89 of the air frame)

Fields are written unconditionally unless noted. Multi-byte types are little-endian.

#### IMU / housekeeping (always present)

| paquete offset | Size | Type | Field | Units / format | Notes |
|----------------|------|------|-------|----------------|-------|
| 0–3 | 4 | `uint32_t` | `rhData` | — | Relative humidity placeholder. Always `0` (WSTK sensor read is disabled). |
| 4–7 | 4 | `int32_t` | `tempData` | — | Temperature placeholder. Always `0` (disabled). |
| 8–11 | 4 | `int32_t` | `accel_x` | g × 10³ | Accelerometer X. Value = raw\_g × 1000. 1 LSB = 0.001 g. ICM20648 at ±2 g full scale. |
| 12–15 | 4 | `int32_t` | `accel_y` | g × 10³ | Accelerometer Y. Same scaling. |
| 16–19 | 4 | `int32_t` | `accel_z` | g × 10³ | Accelerometer Z. Same scaling. |
| 20 | 1 | `int8_t` | `rssi_init` | dBm | RSSI of the first received packet from the parent node. `0` until first reception. |

#### GNSS raw values (written only when `tengo_datos_gps == true`)

If no GNSS data has been received yet, these bytes retain their last value (or zero on boot).

| paquete offset | Size | Type | Field | Units / format | Notes |
|----------------|------|------|-------|----------------|-------|
| 21 | 1 | `uint8_t` | `validez` | ASCII char | GNSS fix validity from NMEA RMC sentence. `'A'` (0x41) = valid fix. `'V'` (0x56) = invalid / no fix. |
| 22–25 | 4 | `float` | `latitud_s` | NMEA DDMM.MMMM | Raw GNSS latitude as parsed from the NMEA GGA sentence. **Not** decimal degrees. To convert: `deg = floor(x/100)`, `min = fmod(x,100)`, `dd = deg + min/60`. |
| 26–29 | 4 | `float` | `latitud_k` | decimal degrees | EKF-estimated latitude. Output of `ENU_to_LLA`. Zero until EKF initialises (≥10 consecutive valid fixes). |
| 30 | 1 | `uint8_t` | `norte_sur` | ASCII char | Hemisphere: `'N'` (0x4E) or `'S'` (0x53). |
| 31–34 | 4 | `float` | `longitud_s` | NMEA DDDMM.MMMM | Raw GNSS longitude (NMEA format, same conversion as latitude). |
| 35–38 | 4 | `float` | `longitud_k` | decimal degrees | EKF-estimated longitude. |
| 39 | 1 | `uint8_t` | `este_oeste` | ASCII char | Hemisphere: `'E'` (0x45) or `'W'` (0x57). |
| 40–43 | 4 | `float` | `altitud_s` | metres (MSL) | Raw GNSS altitude above mean sea level from NMEA GGA. |
| 44–47 | 4 | `float` | `altitud_k` | metres | EKF-estimated altitude. |
| 48–49 | 2 | `uint16_t` | `vel_GNSS_u` | cm/s | GNSS speed over ground. Conversion: `knots / 1.94384 × 100`. 1 LSB = 0.01 m/s. |
| 50–53 | 4 | `float` | `vel_k` | m/s | EKF-estimated 3-D speed magnitude: `√(vE² + vN² + vU²)`. |
| 54–63 | 10 | `char[10]` | `tiempo_out` | `HHMMSS.mmm` | UTC time, ASCII string (no null terminator). GPS-synced: last valid NMEA time + RTC elapsed since fix. `"000000.000"` before first valid fix. |
| 64–69 | 6 | `char[6]` | `fecha_out` | `YYMMDD` | UTC date, ASCII string (no null terminator). Byte-swapped from NMEA `DDMMYY` format. `"000000"` before first valid fix. |
| 70–71 | 2 | `uint16_t` | `PDOP_u` | PDOP × 100 | Position Dilution of Precision × 100. 1 LSB = 0.01. From NMEA GSA field 14. `0` if no GSA received. |

#### EKF / network status (always present)

| paquete offset | Size | Type | Field | Units / format | Notes |
|----------------|------|------|-------|----------------|-------|
| 72 | 1 | `uint8_t` | `GNSS_mode` | 1–7 | Active GNSS constellation mode. 1=GPS, 2=GLONASS, 3=Galileo, 4=DGPS+GPS, 5=GPS+GLONASS, 6=GPS+GLONASS+Galileo, 7=DGPS+GPS+GLONASS+Galileo. |
| 73 | 1 | `char` | `,` | — | Separator byte `0x2C`. |
| 74 | 1 | `char` | `,` | — | Separator byte `0x2C`. |

#### Summary: byte map

```
Byte(s)  Field
──────────────────────────────────────────────────────────
 0– 3    rhData         (uint32_t, always 0)
 4– 7    tempData       (int32_t,  always 0)
 8–11    accel_x        (int32_t,  g×1000)
12–15    accel_y        (int32_t,  g×1000)
16–19    accel_z        (int32_t,  g×1000)
20       rssi_init      (int8_t,   dBm)
21       validez        (uint8_t,  'A' or 'V')
22–25    latitud_s      (float,    NMEA DDMM.MMMM)
26–29    latitud_k      (float,    decimal degrees)
30       norte_sur      (uint8_t,  'N' or 'S')
31–34    longitud_s     (float,    NMEA DDDMM.MMMM)
35–38    longitud_k     (float,    decimal degrees)
39       este_oeste     (uint8_t,  'E' or 'W')
40–43    altitud_s      (float,    metres MSL)
44–47    altitud_k      (float,    metres)
48–49    vel_GNSS_u     (uint16_t, cm/s)
50–53    vel_k          (float,    m/s)
54–63    tiempo_out     (char[10], "HHMMSS.mmm" UTC)
64–69    fecha_out      (char[6],  "YYMMDD" UTC)
70–71    PDOP_u         (uint16_t, PDOP×100)
72       GNSS_mode      (uint8_t,  1–7)
73       ','            (char)
74       ','            (char)
──────────────────────────────────────────────────────────
Total: 75 bytes
```

---

## What does NOT change

- `paquete[75]` byte layout — all `memcpy(paquete+N, ...)` offsets are identical.
- Message header (15 bytes); total frame = 90 bytes.
- `sensorReportPeriodMs` default (250 ms reporting rate).
- `EKF_Predict` / `EKF_Update` function signatures.
- `protocolored.c`, `ekf/cookie_ekf.c`, `ekf/cookie_ekf.h` — no changes.

---

## Verification checklist

1. Build in Simplicity Studio — zero new errors/warnings.
2. Flash sensor node, open serial terminal:
   - `IMU t=...` lines should appear at ~200 Hz (one every ~5 ms).
   - `GP V=A; Mod=...; T=...; D=...; lat=...; lon=...; alt=...; vel=...; COG=...; PDOP=...` lines should appear once per GNSS epoch (~1 Hz when fix active). Prefix is `GN` when using L86 in multi-constellation mode (5–7).
3. With GPS fix held for ≥10 consecutive valid epochs: confirm `ekf_initialized = true`
   (EKF Init print appears once).
4. On radio receiver: confirm packets arrive at 250 ms intervals, packet size = 90 bytes,
   EKF fields (`latitud_k`, `longitud_k`, etc.) non-zero after EKF init.
5. Verify `tiempo_out` in received packet matches UTC time from GNSS within ±1 s.
