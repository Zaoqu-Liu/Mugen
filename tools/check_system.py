#!/usr/bin/env python3
"""Mugen system compatibility checker."""

import platform
import subprocess
import sys
import os


def green(msg):
    return f"\033[32m{msg}\033[0m"


def yellow(msg):
    return f"\033[33m{msg}\033[0m"


def red(msg):
    return f"\033[31m{msg}\033[0m"


def check(label, ok, detail=""):
    mark = green("[PASS]") if ok else red("[FAIL]")
    line = f"  {mark} {label}"
    if detail:
        line += f" ({detail})"
    print(line)
    return ok


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


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
