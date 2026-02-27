#!/usr/bin/env python3

import serial
import glob
import os
import time
from datetime import datetime
from multiprocessing import Process

BAUDRATE = 115200
BYTESIZE = serial.EIGHTBITS
PARITY = serial.PARITY_NONE
STOPBITS = serial.STOPBITS_ONE
TIMEOUT = 1

LOG_DIR = os.path.expanduser("~/inartrans/usb_logs")

def run_timestamp():
    """
    Timestamp used ONCE per process (filename)
    """
    ns = time.time_ns()
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return f"{datetime.fromtimestamp(sec).strftime('%Y-%m-%d_%H-%M-%S')}.{nsec:09d}"

def precise_timestamp():
    """
    Timestamp for each log line
    """
    ns = time.time_ns()
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return f"{datetime.fromtimestamp(sec).strftime('%Y-%m-%d %H:%M:%S')}.{nsec:09d}"

def log_usb(port):
    os.makedirs(LOG_DIR, exist_ok=True)

    # Fixed timestamp for this run
    start_ts = run_timestamp()
    port_name = os.path.basename(port)

    log_file = f"{LOG_DIR}/{port_name}_{start_ts}.log"

    while True:
        try:
            ser = serial.Serial(
                port=port,
                baudrate=BAUDRATE,
                bytesize=BYTESIZE,
                parity=PARITY,
                stopbits=STOPBITS,
                timeout=TIMEOUT
            )

            with open(log_file, "a", buffering=1) as f:
                print(f"[{port}] logging to {log_file}")
                while True:
                    data = ser.readline()
                    if data:
                        line = data.decode(errors="ignore").rstrip()
                        ts = precise_timestamp()
                        f.write(f"[{ts}] {line}\n")

        except Exception as e:
            print(f"[{port}] error: {e} — reconnecting")
            time.sleep(2)

def main():
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    if not ports:
        print("No ttyUSB devices found.")
        return

    processes = []
    for port in ports:
        p = Process(target=log_usb, args=(port,), daemon=True)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()