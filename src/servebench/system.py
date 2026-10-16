"""What machine a run happened on: CPU, RAM and GPUs, read cheaply and without new
dependencies. Every probe is best effort: a missing tool means None, never a failed run.
Parsing is separate from reading so it can be tested with sample output from any OS.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def parse_nvidia_smi(text: str) -> list[str]:
    """`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`
    -> ["NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB, driver 551.76"]"""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        name, mem_mib, driver = parts[0], parts[1], parts[2]
        try:
            mem = f"{int(float(mem_mib)) / 1024:.1f} GB"
        except ValueError:
            mem = mem_mib
        gpus.append(f"{name}, {mem}, driver {driver}")
    return gpus


def parse_meminfo(text: str) -> float | None:
    """Linux /proc/meminfo -> total RAM in GB."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)  # kB -> GB
    return None


def parse_cpuinfo(text: str) -> str | None:
    """Linux /proc/cpuinfo -> CPU model name."""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def gpus() -> list[str] | None:
    if out := _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"]):
        return parse_nvidia_smi(out) or None
    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Apple Silicon: the GPU is part of the chip and shares (unified) memory
        chip = cpu_name()
        return [f"{chip} GPU (unified memory)"] if chip else None
    return None


def cpu_name() -> str | None:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            return out.strip() if out else None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return parse_cpuinfo(f.read())
    except OSError:
        return None


def ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MemStatus()
            m.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return round(m.ullTotalPhys / 2**30, 1)
            return None
        if sys.platform == "darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out) / 2**30, 1) if out else None
        with open("/proc/meminfo", encoding="utf-8") as f:
            return parse_meminfo(f.read())
    except (OSError, ValueError, AttributeError):
        return None


def describe_for_env() -> dict:
    """CPU, RAM and GPUs as Environment fields."""
    return {"cpu": cpu_name(), "ram_gb": ram_gb(), "gpus": gpus()}
