# Sensor-board firmware

Firmware for the "cookie" sensor node — a Silicon Labs EFR32 board carrying an
ICM-20648 IMU and a GNSS receiver, which is the hardware the captured datasets in
`../datasets/` come from. These are Simplicity Studio projects: they are built and
flashed from the IDE, not from this repository's Python tooling.

| Tree | What it is |
|---|---|
| `Inartrans_v2/` | The deployed firmware. IMU + GNSS acquisition, the on-board EKF (`ekf/cookie_ekf.c`), and the Flex/RAIL radio stack. |
| `Inartrans_v2_imu/` | A variant focused on IMU/GNSS fusion; see `CHANGES_IMU_GNSS.md` for how it differs. |
| `Inartrans_porting/` | An in-progress rewrite targeting a current Simplicity Studio, structured into modules under `src/`. Design notes in `docs/`. |

`ekf/cookie_ekf.c` is the embedded counterpart of
[`../scripts/positioning/c/ekf/cookie_ekf.c`](../scripts/positioning/c/ekf/cookie_ekf.c),
so changes to the filter belong in both.

Build products (`*.axf`, `*.hex`, `*.bin`, `*.map`) and generated project metadata are
gitignored.
