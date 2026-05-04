#pragma once

#include <cstdint>
#include <string>

namespace mugen {

struct SystemInfo {
    std::string macos_version;
    std::string macos_build;
    std::string cpu_model;
    uint32_t    cpu_cores_perf    = 0;
    uint32_t    cpu_cores_eff     = 0;
    uint64_t    memory_bytes      = 0;
    std::string gpu_name;
    uint64_t    gpu_memory_bytes  = 0;
    uint64_t    disk_avail_bytes  = 0;
    uint64_t    disk_total_bytes  = 0;
    uint64_t    swap_used_bytes   = 0;
    uint64_t    swap_total_bytes  = 0;
};

auto gather_system_info() -> SystemInfo;

void print_doctor_report(const SystemInfo& info);

}  // namespace mugen
