#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mugen {

struct SystemState {
    size_t total_memory = 0;
    size_t available_memory = 0;
    size_t mugen_used_memory = 0;
    float memory_pressure = 0.0f;

    size_t ssd_available_bytes = 0;

    float gpu_utilization = 0.0f;

    float cpu_temp_celsius = 0.0f;
    float power_watts = 0.0f;

    uint64_t timestamp = 0;
};

enum class DegradationLevel : uint8_t {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Emergency = 3,
};

class SystemMonitor {
public:
    struct Config {
        float warning_threshold = 0.3f;
        float critical_threshold = 0.7f;
        float emergency_threshold = 0.9f;
        uint32_t poll_interval_ms = 500;
        size_t min_ssd_bytes = 1ULL << 30;
    };

    SystemMonitor();
    explicit SystemMonitor(Config config);
    ~SystemMonitor();

    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    void start();
    void stop();

    auto current_state() const -> SystemState;
    auto degradation_level() const -> DegradationLevel;

    using DegradationCallback = std::function<void(DegradationLevel old_level,
                                                   DegradationLevel new_level)>;
    void on_degradation_change(DegradationCallback callback);

    void force_degradation(DegradationLevel level);

    auto check_ssd_space(size_t required_bytes) const -> bool;

    auto format_metrics(float tok_per_sec, float cache_hit_rate,
                        uint32_t prefetch_depth) const -> std::string;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Logger {
public:
    enum class Level { Debug, Info, Warn, Error };

    static auto instance() -> Logger&;

    void set_log_dir(const std::filesystem::path& dir);
    void set_level(Level level);

    void log(Level level, const std::string& component,
             const std::string& message);

    void debug(const std::string& component, const std::string& msg);
    void info(const std::string& component, const std::string& msg);
    void warn(const std::string& component, const std::string& msg);
    void error(const std::string& component, const std::string& msg);

    void flush();

    ~Logger();

private:
    Logger();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mugen
