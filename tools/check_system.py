#!/usr/bin/env python3
"""Mugen system compatibility checker."""

import platform
import subprocess
import sys
import os
import re
import plistlib


def green(msg):
    return f"\033[32m{msg}\033[0m"


def yellow(msg):
    return f"\033[33m{msg}\033[0m"


def red(msg):
    return f"\033[31m{msg}\033[0m"


def bold(msg):
    return f"\033[1m{msg}\033[0m"


def check(label, ok, detail=""):
    mark = green("[PASS]") if ok else red("[FAIL]")
    line = f"  {mark} {label}"
    if detail:
        line += f" ({detail})"
    print(line)
    return ok


def info(label, detail=""):
    line = f"  [INFO] {label}"
    if detail:
        line += f" ({detail})"
    print(line)


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


# -------------------------------------------------------------------------
# Storage diagnostics
# -------------------------------------------------------------------------

def check_storage_type():
    """Detect NVMe vs SATA and estimate sequential read speed."""
    raw = run_cmd("system_profiler SPNVMeDataType 2>/dev/null")
    if raw:
        ssd_type = "NVMe"
        speed_est = "~5,000-7,400 MB/s (Apple NVMe)"
        m = re.search(r"Link Width:\s*x(\d+)", raw)
        lanes = int(m.group(1)) if m else 4
        if lanes >= 4:
            speed_est = f"~5,000-7,400 MB/s (NVMe x{lanes})"
        else:
            speed_est = f"~2,000-3,500 MB/s (NVMe x{lanes})"
    else:
        raw_sata = run_cmd("system_profiler SPSerialATADataType 2>/dev/null")
        if raw_sata and "SATA" in raw_sata:
            ssd_type = "SATA"
            speed_est = "~500-560 MB/s (SATA III)"
        else:
            ssd_type = "Unknown"
            speed_est = "unable to detect"

    return ssd_type, speed_est


# -------------------------------------------------------------------------
# Metal GPU diagnostics
# -------------------------------------------------------------------------

def get_metal_gpu_cores():
    """Return the GPU core count from the chip identifier."""
    chip = run_cmd("sysctl -n machdep.cpu.brand_string")
    if not chip:
        return 0, "unknown chip"

    gpu_cores = 0
    m = re.search(r"Apple\s+(M\d+)\s*(Pro|Max|Ultra)?", chip)
    if m:
        gen = m.group(1)
        variant = m.group(2) or "base"
        core_map = {
            ("M1", "base"): 8, ("M1", "Pro"): 16, ("M1", "Max"): 32, ("M1", "Ultra"): 64,
            ("M2", "base"): 10, ("M2", "Pro"): 19, ("M2", "Max"): 38, ("M2", "Ultra"): 76,
            ("M3", "base"): 10, ("M3", "Pro"): 18, ("M3", "Max"): 40, ("M3", "Ultra"): 80,
            ("M4", "base"): 10, ("M4", "Pro"): 20, ("M4", "Max"): 40, ("M4", "Ultra"): 80,
        }
        gpu_cores = core_map.get((gen, variant), 0)

    return gpu_cores, chip


def get_memory_bandwidth():
    """Estimate unified memory bandwidth based on chip variant."""
    chip = run_cmd("sysctl -n machdep.cpu.brand_string")
    m = re.search(r"Apple\s+(M\d+)\s*(Pro|Max|Ultra)?", chip)
    if not m:
        return 0, "unknown"

    gen = m.group(1)
    variant = m.group(2) or "base"
    bw_map = {
        ("M1", "base"): 68, ("M1", "Pro"): 200, ("M1", "Max"): 400, ("M1", "Ultra"): 800,
        ("M2", "base"): 100, ("M2", "Pro"): 200, ("M2", "Max"): 400, ("M2", "Ultra"): 800,
        ("M3", "base"): 100, ("M3", "Pro"): 150, ("M3", "Max"): 400, ("M3", "Ultra"): 800,
        ("M4", "base"): 120, ("M4", "Pro"): 273, ("M4", "Max"): 546, ("M4", "Ultra"): 819,
    }
    bw = bw_map.get((gen, variant), 0)
    return bw, f"{gen} {variant}"


# -------------------------------------------------------------------------
# Model recommendation
# -------------------------------------------------------------------------

def recommend_model(mem_gb, bw_gb_s):
    """Suggest the largest runnable model based on available unified memory and bandwidth."""
    print()
    print(bold("Model Recommendation"))
    print("-" * 50)

    usable_gb = mem_gb * 0.85

    recommendations = [
        (130, "DeepSeek-V3 671B Q4_K (full, ~360 GB)", 360),
        (96,  "Llama-3.1 405B Q3_K (~200 GB)", 200),
        (64,  "DeepSeek-V3 671B Q2_K (~170 GB)", 170),
        (48,  "Llama-3.1 70B Q4_K (~40 GB)", 40),
        (32,  "Qwen2.5 32B Q4_K (~20 GB)", 20),
        (24,  "Llama-3.1 8B Q8_0 (~8 GB)", 8),
        (16,  "Llama-3.1 8B Q4_K (~5 GB)", 5),
        (8,   "Qwen2.5 3B Q4_K (~2 GB)", 2),
    ]

    recommended = None
    for threshold, label, model_gb in recommendations:
        if usable_gb >= model_gb:
            recommended = (label, model_gb)
            break

    if recommended:
        label, model_gb = recommended
        print(f"  Max recommended model: {green(label)}")
        print(f"  Memory budget:         {usable_gb:.0f} GB usable / {mem_gb} GB total")
        if bw_gb_s > 0:
            q4_bytes_per_param = 0.5625
            est_tps = bw_gb_s / (model_gb if model_gb > 0 else 1)
            print(f"  Est. decode throughput: ~{est_tps:.0f} tok/s (bandwidth-limited estimate)")
    else:
        print(f"  {red('Insufficient memory')} ({mem_gb} GB) for any supported model.")
        print("  Minimum: 8 GB for small 3B models.")


def main():
    print("Mugen System Compatibility Check")
    print("=" * 50)
    all_ok = True

    # macOS version
    ver = platform.mac_ver()[0]
    if ver:
        parts = ver.split(".")
        major = int(parts[0]) if parts else 0
        all_ok &= check("macOS version >= 14", major >= 14, f"macOS {ver}")
    else:
        all_ok &= check("macOS", False, "not running on macOS")

    # Apple Silicon
    arch = platform.machine()
    all_ok &= check("Apple Silicon (arm64)", arch == "arm64", arch)

    # Unified memory
    mem_bytes = run_cmd("sysctl -n hw.memsize")
    mem_gb = 0
    if mem_bytes:
        mem_gb = int(mem_bytes) // (1024 ** 3)
        all_ok &= check("Unified memory >= 32 GB", mem_gb >= 32, f"{mem_gb} GB")
    else:
        all_ok &= check("Unified memory", False)

    # Xcode CLT
    xcode_path = run_cmd("xcode-select -p")
    has_xcode = bool(xcode_path and os.path.exists(xcode_path))
    all_ok &= check("Xcode Command Line Tools", has_xcode, xcode_path if has_xcode else "not found")

    # Metal compiler
    metal_path = run_cmd("xcrun --find metal 2>/dev/null")
    has_metal = bool(metal_path and "error" not in metal_path.lower())
    all_ok &= check("Metal compiler (Xcode.app)", has_metal,
                    metal_path if has_metal else "requires full Xcode, not just CLT")

    # CMake
    cmake_ver = run_cmd("cmake --version 2>/dev/null | head -1")
    if cmake_ver:
        ver_str = cmake_ver.split()[-1]
        parts = ver_str.split(".")
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        all_ok &= check("CMake >= 3.21", major > 3 or (major == 3 and minor >= 21), ver_str)
    else:
        all_ok &= check("CMake", False, "not found")

    # Available disk space (models dir)
    home = os.path.expanduser("~")
    models_dir = os.environ.get("MUGEN_MODEL_DIR", os.path.join(home, ".mugen", "models"))
    parent = os.path.dirname(models_dir)
    if os.path.exists(parent):
        stat = os.statvfs(parent)
        free_gb = (stat.f_frsize * stat.f_bavail) // (1024 ** 3)
        all_ok &= check("Disk space >= 100 GB (for models)", free_gb >= 100,
                        f"{free_gb} GB free on {parent}")
    else:
        all_ok &= check("Disk space", False, f"{parent} does not exist")

    # --- Extended hardware diagnostics ---
    print()
    print(bold("Extended Hardware Info"))
    print("-" * 50)

    # SSD type and speed
    ssd_type, speed_est = check_storage_type()
    info(f"Storage: {ssd_type}", speed_est)

    # Metal GPU cores
    gpu_cores, chip_name = get_metal_gpu_cores()
    if gpu_cores > 0:
        info(f"Metal GPU cores: {gpu_cores}", chip_name)
    else:
        info("Metal GPU cores: unable to detect", chip_name)

    # Memory bandwidth
    bw, chip_label = get_memory_bandwidth()
    if bw > 0:
        info(f"Memory bandwidth: ~{bw} GB/s", chip_label)
    else:
        info("Memory bandwidth: unable to estimate", chip_label)

    # Model recommendation
    recommend_model(mem_gb, bw)

    print()
    print("=" * 50)
    if all_ok:
        print(green("All checks passed. Your system is ready for Mugen."))
    else:
        print(red("Some checks failed. See above for details."))
        print("Note: Metal compiler requires full Xcode (not just Command Line Tools).")
        print("Install from: https://developer.apple.com/xcode/")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
