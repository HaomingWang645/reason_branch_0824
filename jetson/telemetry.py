"""Telemetry sampler for Jetson AGX Orin system measurements.

Samples INA3221 power rails (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0), RAM,
and thermal zones in a background thread; integrates power -> energy over
labeled measurement windows.

Usage:
    tel = Telemetry(hz=20)
    tel.start()
    with tel.window("prefill_b3"):
        ...work...
    tel.stop()
    tel.report()  # list of dicts per window
"""
import threading
import time
from contextlib import contextmanager
from pathlib import Path

HWMON = Path("/sys/class/hwmon/hwmon1")  # ina3221 on AGX Orin devkit
RAILS = {1: "VDD_GPU_SOC", 2: "VDD_CPU_CV", 3: "VIN_SYS_5V0"}
MEMINFO = Path("/proc/meminfo")
THERMAL = {
    "gpu": Path("/sys/devices/virtual/thermal/thermal_zone1/temp"),
    "tj": Path("/sys/devices/virtual/thermal/thermal_zone5/temp"),
}


def _read_int(p: Path) -> int:
    try:
        return int(p.read_text())
    except Exception:
        return 0


def read_power_mw() -> dict:
    out = {}
    for ch, name in RAILS.items():
        v = _read_int(HWMON / f"in{ch}_input")     # mV
        i = _read_int(HWMON / f"curr{ch}_input")   # mA
        out[name] = v * i / 1000.0                 # mW
    return out


def read_mem_used_mb() -> float:
    total = avail = 0
    for line in MEMINFO.read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail = int(line.split()[1])
            break
    return (total - avail) / 1024.0


def read_temps_c() -> dict:
    return {k: _read_int(p) / 1000.0 for k, p in THERMAL.items()}


class Telemetry:
    def __init__(self, hz: float = 20.0):
        self.period = 1.0 / hz
        self.samples = []  # (t, {rail: mW}, mem_mb, {zone: C})
        self._run = False
        self._thread = None
        self.windows = []  # (label, t0, t1)

    def start(self):
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._run:
            t = time.monotonic()
            self.samples.append((t, read_power_mw(), read_mem_used_mb(), read_temps_c()))
            dt = time.monotonic() - t
            if dt < self.period:
                time.sleep(self.period - dt)

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2)

    @contextmanager
    def window(self, label: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            t1 = time.monotonic()
            self.windows.append((label, t0, t1))

    def _window_stats(self, label, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        dur = t1 - t0
        rec = {"label": label, "latency_s": dur, "n_samples": len(s)}
        if s:
            for rail in RAILS.values():
                p = [x[1][rail] for x in s]
                avg_w = sum(p) / len(p) / 1000.0
                rec[f"{rail}_avg_w"] = avg_w
                rec[f"{rail}_energy_j"] = avg_w * dur
            rec["mem_peak_mb"] = max(x[2] for x in s)
            rec["gpu_temp_max_c"] = max(x[3].get("gpu", 0) for x in s)
        return rec

    def report(self):
        return [self._window_stats(*w) for w in self.windows]

    def baseline_idle_w(self, seconds: float = 5.0):
        """Measure idle power before the run; call while system is quiet."""
        t0 = time.monotonic()
        vals = []
        while time.monotonic() - t0 < seconds:
            vals.append(sum(read_power_mw().values()) / 1000.0)
            time.sleep(0.05)
        return sum(vals) / len(vals)
