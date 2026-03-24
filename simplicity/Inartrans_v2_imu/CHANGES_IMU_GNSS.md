# IMU and GNSS Improvements

## Summary

Replaced polled/timer-based sampling with hardware-interrupt-driven approaches for both IMU (200 Hz) and GNSS (event-driven on sentence arrival). Only serial print — no radio transmission, no flash storage.

---

## Serial Output Format

```
IMU t=<ms>; A=<ax>,<ay>,<az>; G=<gx>,<gy>,<gz>
GP V=<A|V>; T=<ms_midnight>; D=<DDMMYY>; Lat=<lat*10000>; Lon=<lon*10000>; Alt=<alt*100>; Vel=<m/s*100>; COG=<deg*100>; PDOP=<pdop*100>
GN V=<A|V>; T=<ms_midnight>; D=<DDMMYY>; Lat=<lat*10000>; Lon=<lon*10000>; Alt=<alt*100>; Vel=<m/s*100>; COG=<deg*100>; PDOP=<pdop*100>
```

The prefix (`GP` or `GN`) is auto-detected from the NMEA talker ID in each epoch — no manual configuration needed.

- **t / T**: milliseconds since UTC midnight (see timestamp section below)
- **A**: accelerometer ×1000 (raw g-force × 1000, integer)
- **G**: gyroscope ×100 (deg/s × 100, integer)
- **Lat/Lon**: degrees × 10000 (integer)
- **Alt**: meters × 100 (integer)
- **Vel**: m/s × 100 (converted from knots)
- **COG**: course over ground, degrees × 100
- **PDOP**: position dilution of precision × 100

---

## Timestamp and Time Variables

### Clock source

The RTCC peripheral is clocked from LFRCO at 32768 Hz. RTCDRV configures a /8 prescaler, giving a tick rate of **4096 Hz** (confirmed in `main.c`: "4 ticks es un milisegundo"). Dividing raw ticks by 4 converts to milliseconds with ~2.3% error (actual: 4.096 ticks/ms, so each unit = 0.9766 ms).

### Variables (`flex-callbacks.c`)

| Variable | Type | Description |
|---|---|---|
| `gps_ms_ref` | `uint32_t` | GPS time of last valid fix, **exact milliseconds since UTC midnight** (from NMEA "HHMMSS.sss" parse) |
| `rtc_at_gps_fix` | `uint32_t` | Raw RTCDRV tick value at the moment the last GPS fix was parsed |
| `gps_time_valid` | `bool` | `false` until the first valid NMEA epoch is received |

### Timestamp computation (IMU line)

```c
uint32_t rtc_now = RTCDRV_GetWallClockTicks32();   // raw ticks at ~4096 Hz
uint32_t ts = gps_time_valid
              ? (gps_ms_ref + (rtc_now - rtc_at_gps_fix) / 4)  // GPS-synced ms since midnight
              : (rtc_now / 4);                                   // ms since boot (fallback)
```

- **GPS-synced mode** (`gps_time_valid == true`): `ts` is milliseconds since UTC midnight.
  `gps_ms_ref` is exact; the RTC-elapsed term has ~2.3% error (23 ms per second of drift from last fix).
- **Fallback mode** (`gps_time_valid == false`): `ts` is milliseconds since device boot (same ~2.3% error).

### GNSS `T=` field

`T=` prints `gps_ms_ref` directly — this is **exact milliseconds since UTC midnight**, parsed from the NMEA time field. No RTC approximation involved.

### Overflow limits

- `RTCDRV_GetWallClockTicks32()` overflows at 2³²/4096 = **1,048,576 s ≈ 12.1 days**. The unsigned subtraction `(rtc_now - rtc_at_gps_fix)` handles a single wrap correctly, but breaks if more than 12.1 days pass without a GPS fix.
- `gps_ms_ref` max = 86,399,999 ms (end of day). `ts` overflows `uint32_t` (4.29B) only after ~49 days of accumulated drift — not a practical concern.

### Converting `ts` to readable time (Python)

```python
ts_ms = 52407456  # example value from serial log

h  = ts_ms // 3_600_000
m  = (ts_ms % 3_600_000) // 60_000
s  = (ts_ms % 60_000)    // 1_000
ms = ts_ms % 1_000
print(f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}")  # → "14:33:27.456"

# Full UTC datetime (today's date):
from datetime import datetime, timezone, timedelta
epoch = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
t = epoch + timedelta(milliseconds=ts_ms)
print(t.isoformat())  # → "2026-03-24T14:33:27.456000+00:00"
```

---

## Changes Made

### 1. `main/EFR32/main.c`
- `IMU_config(20)` → `IMU_config(200)` — sensor now outputs at 200 Hz instead of 20 Hz.
  - **Bug**: code was trying to read at 200 Hz but sensor only produced new data at 20 Hz → 90% of reads returned stale values.

### 2. `imu.c`
- Bandwidth filter: `ICM20648_ACCEL_BW_6HZ` / `ICM20648_GYRO_BW_6HZ` → `ICM20648_ACCEL_BW_24HZ` / `ICM20648_GYRO_BW_24HZ`

| Filter BW | Phase lag @ 200 Hz | Notes |
|-----------|-------------------|-------|
| 6 Hz (old) | ~83 ms | Kills all dynamics above walking pace; useless for EKF |
| **24 Hz (new)** | **~21 ms** | Removes vibration/spikes, preserves navigation dynamics (0–10 Hz) |
| 50 Hz | ~10 ms | Better for fast platforms (drones, bikes) |

### 3. `hal/com.c` — USART1 RX ISR
- Added `gnss_lines_received++` on each `\n` (end of NMEA sentence).
- Added buffer overflow protection: if `indice >= MI_BUFFER_SIZE - 1`, reset buffer and counters.
- **Bugs fixed**:
  - Buffer was never reset on overflow → silent byte drops, corrupted epochs.
  - No way to know when a complete NMEA epoch had arrived.

### 4. `flex-callbacks.c`

#### IMU — 200 Hz via RTCDRV periodic timer
- `IMU_initTimer()`: allocates and starts an RTCDRV periodic timer at 5 ms (= 200 Hz).
- ISR callback sets `volatile bool imu_sample_pending = true`.
- `emberAfMainTickCallback()` checks the flag each main-loop iteration, reads accel + gyro via I2C, and prints immediately.
- I2C read stays in main context (never in ISR) — safe.

#### GNSS — event-driven on epoch arrival
- `emberAfMainTickCallback()` checks `gnss_lines_received >= 6`.
  - 6 lines ensures RMC + GGA + GSA are all in the buffer (9600 baud, ~500 ms/epoch).
- Calls `busca2()` to parse the buffer.
- Auto-detects NMEA talker ID (`GP` vs `GN`) by scanning the first `$G?` sentence in each epoch buffer. Prints the detected prefix instead of a hardcoded label, so L80 (GPS-only) prints `GP` and L86 multi-constellation prints `GN`.
- Prints data immediately on successful parse.
- Buffer reset strategy: only resets when buffer >75% full. `busca2()` resets internally on a successful parse. This allows the buffer to accumulate across two epochs when a GNSS module outputs GGA before RMC (non-default ordering), while still protecting against overflow on multi-constellation modules (L86).
- PDOP-based GNSS mode selector runs here (on each received epoch) instead of on the fixed timer.
- **Bugs fixed**:
  - Old code called `busca2()` on a 125 ms timer → partial buffer, parse failures, buffer accumulation.
  - `tengo_gps` was set to 1 on first byte and never cleared.

#### `reportHandler` — simplified
- Removed IMU sampling block.
- Removed GPS processing block.
- Now only advances the GNSS constellation search counter (~8 Hz via `sensorReportPeriodMs`).
- **Added `emberEventControlSetDelayMS(reportControl, sensorReportPeriodMs)`** — re-arms the Ember event each call (was missing before; event only fired once after boot).

---

## Notes

- `GNSS_LINES_PER_EPOCH = 6` works for both Quectel L80 (GPS-only, ~7 lines/epoch) and L86 (multi-constellation, 12–20 lines/epoch). Both modules output RMC first by default (Quectel PMTK default), so RMC+GGA+GSA are all present within the first 6 sentences.
- If your module is configured GGA-first, the buffer accumulation strategy (overflow-only reset) handles it — the parser will find RMC from epoch N and GGA from epoch N+1 in the same buffer.
- IMU bandwidth can be changed to `ICM20648_ACCEL_BW_50HZ` / `ICM20648_GYRO_BW_51HZ` for faster platforms (drones, bikes).
- All serial output goes to VCOM (USB CDC), not to the GPS UART (USART1 at 9600 baud).
