#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "cli/doctor.h"

#include <cstdio>
#include <cstring>
#include <sys/mount.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/mach.h>

namespace mugen {

namespace {

auto sysctl_string(const char* name) -> std::string {
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname(name, buf, &len, nullptr, 0) == 0)
        return {buf, len > 0 ? len - 1 : 0};
    return {};
}

auto sysctl_u32(const char* name) -> uint32_t {
    uint32_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, nullptr, 0);
    return val;
}

auto sysctl_u64(const char* name) -> uint64_t {
    uint64_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, nullptr, 0);
    return val;
}

auto query_gpu_info(std::string& name_out, uint64_t& mem_out) -> bool {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return false;
        name_out = std::string([[device name] UTF8String]);
        mem_out  = static_cast<uint64_t>([device recommendedMaxWorkingSetSize]);
        return true;
    }
}

auto query_disk_space(uint64_t& avail_out, uint64_t& total_out) -> bool {
    struct statfs fs;
    if (statfs("/", &fs) != 0) return false;
    avail_out = static_cast<uint64_t>(fs.f_bavail) * static_cast<uint64_t>(fs.f_bsize);
    total_out = static_cast<uint64_t>(fs.f_blocks) * static_cast<uint64_t>(fs.f_bsize);
    return true;
}

auto query_swap_usage(uint64_t& used_out, uint64_t& total_out) -> bool {
    int mib[2] = {CTL_VM, VM_SWAPUSAGE};
    struct xsw_usage sw = {};
    size_t len = sizeof(sw);
    if (sysctl(mib, 2, &sw, &len, nullptr, 0) != 0) return false;
    used_out  = sw.xsu_used;
    total_out = sw.xsu_total;
    return true;
}

auto query_macos_version(std::string& version_out, std::string& build_out) -> void {
    @autoreleasepool {
        NSProcessInfo* pi = [NSProcessInfo processInfo];
        NSOperatingSystemVersion v = [pi operatingSystemVersion];
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%ld.%ld.%ld",
                      (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion);
        version_out = buf;
    }
    build_out = sysctl_string("kern.osversion");
}

auto format_bytes(uint64_t bytes) -> std::string {
    char buf[64];
    if (bytes >= (1ULL << 40)) {
        std::snprintf(buf, sizeof(buf), "%.1f TB", static_cast<double>(bytes) / (1ULL << 40));
    } else if (bytes >= (1ULL << 30)) {
        std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / (1ULL << 30));
    } else if (bytes >= (1ULL << 20)) {
        std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1ULL << 20));
    } else {
        std::snprintf(buf, sizeof(buf), "%llu bytes", static_cast<unsigned long long>(bytes));
    }
    return buf;
}

}  // namespace

auto gather_system_info() -> SystemInfo {
    SystemInfo info;

    query_macos_version(info.macos_version, info.macos_build);

    info.cpu_model = sysctl_string("machdep.cpu.brand_string");
    if (info.cpu_model.empty())
        info.cpu_model = sysctl_string("hw.model");

    info.cpu_cores_perf = sysctl_u32("hw.perflevel0.physicalcpu");
    info.cpu_cores_eff  = sysctl_u32("hw.perflevel1.physicalcpu");

    info.memory_bytes = sysctl_u64("hw.memsize");

    query_gpu_info(info.gpu_name, info.gpu_memory_bytes);
    query_disk_space(info.disk_avail_bytes, info.disk_total_bytes);
    query_swap_usage(info.swap_used_bytes, info.swap_total_bytes);

    return info;
}

void print_doctor_report(const SystemInfo& info) {
    constexpr int kLabelWidth = 22;

    std::printf("\n");
    std::printf("  Mugen Environment Diagnostics\n");
    std::printf("  =============================\n\n");

    std::printf("  %-*s  macOS %s (%s)\n", kLabelWidth, "OS",
                info.macos_version.c_str(), info.macos_build.c_str());

    std::printf("  %-*s  %s\n", kLabelWidth, "CPU", info.cpu_model.c_str());

    if (info.cpu_cores_perf > 0 || info.cpu_cores_eff > 0) {
        std::printf("  %-*s  %uP + %uE\n", kLabelWidth, "CPU Cores",
                    info.cpu_cores_perf, info.cpu_cores_eff);
    }

    std::printf("  %-*s  %s\n", kLabelWidth, "Memory (unified)",
                format_bytes(info.memory_bytes).c_str());

    if (!info.gpu_name.empty()) {
        std::printf("  %-*s  %s\n", kLabelWidth, "GPU", info.gpu_name.c_str());
        std::printf("  %-*s  %s\n", kLabelWidth, "GPU Working Set",
                    format_bytes(info.gpu_memory_bytes).c_str());
    } else {
        std::printf("  %-*s  [not detected]\n", kLabelWidth, "GPU");
    }

    std::printf("  %-*s  %s / %s\n", kLabelWidth, "Disk (available)",
                format_bytes(info.disk_avail_bytes).c_str(),
                format_bytes(info.disk_total_bytes).c_str());

    if (info.swap_total_bytes > 0) {
        std::printf("  %-*s  %s / %s\n", kLabelWidth, "Swap",
                    format_bytes(info.swap_used_bytes).c_str(),
                    format_bytes(info.swap_total_bytes).c_str());
    } else {
        std::printf("  %-*s  disabled\n", kLabelWidth, "Swap");
    }

    std::printf("\n");

    // Recommendations
    bool has_warning = false;

    if (info.memory_bytes < 16ULL * (1ULL << 30)) {
        if (!has_warning) {
            std::printf("  Warnings\n");
            std::printf("  --------\n");
            has_warning = true;
        }
        std::printf("  [!] %s RAM detected. MoE models (e.g. DeepSeek V3) require\n",
                    format_bytes(info.memory_bytes).c_str());
        std::printf("      32 GB+ for good performance. Consider smaller quantizations.\n");
    }

    if (info.disk_avail_bytes < 20ULL * (1ULL << 30)) {
        if (!has_warning) {
            std::printf("  Warnings\n");
            std::printf("  --------\n");
            has_warning = true;
        }
        std::printf("  [!] Only %s disk space available. Large models can be 50-100 GB.\n",
                    format_bytes(info.disk_avail_bytes).c_str());
        std::printf("      Free up space or change model directory with MUGEN_MODEL_DIR.\n");
    }

    if (info.gpu_name.empty()) {
        if (!has_warning) {
            std::printf("  Warnings\n");
            std::printf("  --------\n");
            has_warning = true;
        }
        std::printf("  [!] No Metal GPU detected. Mugen requires Apple Silicon with Metal.\n");
    }

    if (info.swap_used_bytes > 2ULL * (1ULL << 30)) {
        if (!has_warning) {
            std::printf("  Warnings\n");
            std::printf("  --------\n");
            has_warning = true;
        }
        std::printf("  [!] High swap usage (%s). This may degrade inference performance.\n",
                    format_bytes(info.swap_used_bytes).c_str());
        std::printf("      Close other applications to free memory.\n");
    }

    if (!has_warning) {
        std::printf("  Status: all checks passed.\n");
    }

    std::printf("\n");
}

}  // namespace mugen
