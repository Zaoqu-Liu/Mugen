#include "core/monitor/system_monitor.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>

#import <Foundation/Foundation.h>

namespace mugen {

// ===========================================================================
// SystemMonitor::Impl
// ===========================================================================

struct SystemMonitor::Impl {
    Config config;

    mutable std::mutex state_mutex;
    SystemState state{};
    std::atomic<DegradationLevel> level{DegradationLevel::Normal};

    std::mutex callback_mutex;
    std::vector<DegradationCallback> callbacks;

    std::thread poll_thread;
    std::mutex poll_mutex;
    std::condition_variable poll_cv;
    bool running = false;

    size_t total_physical_memory = 0;

    explicit Impl(Config cfg) : config(cfg) {
        total_physical_memory = query_total_memory();
        poll_once();
    }

    static auto query_total_memory() -> size_t {
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        uint64_t mem = 0;
        size_t len = sizeof(mem);
        sysctl(mib, 2, &mem, &len, nullptr, 0);
        return static_cast<size_t>(mem);
    }

    auto query_available_memory() -> size_t {
        vm_statistics64_data_t vm_stat{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vm_stat),
                              &count) != KERN_SUCCESS) {
            return 0;
        }

        vm_size_t page_size = 0;
        host_page_size(mach_host_self(), &page_size);

        // free + inactive + purgeable ≈ memory the system can reclaim
        uint64_t reclaimable = static_cast<uint64_t>(vm_stat.free_count)
                             + vm_stat.inactive_count
                             + vm_stat.purgeable_count;
        return static_cast<size_t>(reclaimable * page_size);
    }

    static auto query_process_memory() -> size_t {
        mach_task_basic_info_data_t info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&info),
                      &count) != KERN_SUCCESS) {
            return 0;
        }
        return static_cast<size_t>(info.resident_size);
    }

    static auto query_ssd_available() -> size_t {
        @autoreleasepool {
            NSString* home = NSHomeDirectory();
            NSDictionary* attrs = [[NSFileManager defaultManager]
                attributesOfFileSystemForPath:home
                                       error:nil];
            if (attrs) {
                NSNumber* freeSize = attrs[NSFileSystemFreeSize];
                return static_cast<size_t>([freeSize unsignedLongLongValue]);
            }
        }
        struct statvfs st{};
        if (statvfs("/", &st) == 0)
            return static_cast<size_t>(st.f_bavail) * st.f_frsize;
        return 0;
    }

    void poll_once() {
        SystemState s;
        s.total_memory = total_physical_memory;
        s.available_memory = query_available_memory();
        s.mugen_used_memory = query_process_memory();
        s.memory_pressure = (s.total_memory > 0)
            ? 1.0f - static_cast<float>(s.available_memory)
                    / static_cast<float>(s.total_memory)
            : 0.0f;
        s.memory_pressure = std::clamp(s.memory_pressure, 0.0f, 1.0f);

        s.ssd_available_bytes = query_ssd_available();

        // No reliable public API for GPU util, CPU temp, or power on macOS
        s.gpu_utilization = 0.0f;
        s.cpu_temp_celsius = 0.0f;
        s.power_watts = 0.0f;

        auto now = std::chrono::steady_clock::now();
        s.timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count());

        DegradationLevel new_level = compute_level(s.memory_pressure);
        DegradationLevel old_level = level.load(std::memory_order_relaxed);

        {
            std::lock_guard lock(state_mutex);
            state = s;
        }

        if (new_level != old_level) {
            level.store(new_level, std::memory_order_release);
            fire_callbacks(old_level, new_level);
        }
    }

    auto compute_level(float pressure) const -> DegradationLevel {
        if (pressure >= config.emergency_threshold)
            return DegradationLevel::Emergency;
        if (pressure >= config.critical_threshold)
            return DegradationLevel::Critical;
        if (pressure >= config.warning_threshold)
            return DegradationLevel::Warning;
        return DegradationLevel::Normal;
    }

    void fire_callbacks(DegradationLevel old_lvl, DegradationLevel new_lvl) {
        std::lock_guard lock(callback_mutex);
        for (auto& cb : callbacks) {
            if (cb) cb(old_lvl, new_lvl);
        }
    }

    void run() {
        while (true) {
            std::unique_lock lock(poll_mutex);
            if (poll_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(config.poll_interval_ms),
                    [this] { return !running; })) {
                break;
            }
            poll_once();
        }
    }
};

// ---------------------------------------------------------------------------
// SystemMonitor public API
// ---------------------------------------------------------------------------

SystemMonitor::SystemMonitor() : SystemMonitor(Config{}) {}

SystemMonitor::SystemMonitor(Config config)
    : impl_(std::make_unique<Impl>(config)) {}

SystemMonitor::~SystemMonitor() { stop(); }

void SystemMonitor::start() {
    {
        std::lock_guard lock(impl_->poll_mutex);
        if (impl_->running) return;
        impl_->running = true;
    }
    impl_->poll_thread = std::thread([this] { impl_->run(); });
}

void SystemMonitor::stop() {
    {
        std::lock_guard lock(impl_->poll_mutex);
        if (!impl_->running) return;
        impl_->running = false;
    }
    impl_->poll_cv.notify_all();
    if (impl_->poll_thread.joinable())
        impl_->poll_thread.join();
}

auto SystemMonitor::current_state() const -> SystemState {
    std::lock_guard lock(impl_->state_mutex);
    return impl_->state;
}

auto SystemMonitor::degradation_level() const -> DegradationLevel {
    return impl_->level.load(std::memory_order_acquire);
}

void SystemMonitor::on_degradation_change(DegradationCallback callback) {
    std::lock_guard lock(impl_->callback_mutex);
    impl_->callbacks.push_back(std::move(callback));
}

void SystemMonitor::force_degradation(DegradationLevel level) {
    DegradationLevel old_level =
        impl_->level.load(std::memory_order_relaxed);
    if (old_level != level) {
        impl_->level.store(level, std::memory_order_release);
        impl_->fire_callbacks(old_level, level);
    }
}

auto SystemMonitor::check_ssd_space(size_t required_bytes) const -> bool {
    std::lock_guard lock(impl_->state_mutex);
    return impl_->state.ssd_available_bytes >= required_bytes;
}

auto SystemMonitor::format_metrics(float tok_per_sec, float cache_hit_rate,
                                   uint32_t prefetch_depth) const
    -> std::string {
    auto s = current_state();
    auto lvl = degradation_level();

    constexpr float kGB = 1024.0f * 1024.0f * 1024.0f;
    float used_gb =
        static_cast<float>(s.total_memory - s.available_memory) / kGB;
    float total_gb = static_cast<float>(s.total_memory) / kGB;

    const char* mode_str = nullptr;
    switch (lvl) {
        case DegradationLevel::Normal:    mode_str = "normal";    break;
        case DegradationLevel::Warning:   mode_str = "warning";   break;
        case DegradationLevel::Critical:  mode_str = "critical";  break;
        case DegradationLevel::Emergency: mode_str = "emergency"; break;
    }

    char buf[256];
    std::snprintf(
        buf, sizeof(buf),
        "[Mugen] %.1f tok/s | cache: %.1f%% | mem: %.1f/%.1f GB"
        " | prefetch: %u | mode: %s",
        tok_per_sec, cache_hit_rate * 100.0f, used_gb, total_gb,
        prefetch_depth, mode_str);

    return {buf};
}

// ===========================================================================
// Logger::Impl
// ===========================================================================

struct Logger::Impl {
    std::mutex mutex;
    Level min_level = Level::Info;
    std::filesystem::path log_dir;
    std::ofstream current_file;
    std::string current_date;

    Impl() {
        const char* home = std::getenv("HOME");
        log_dir =
            std::filesystem::path(home ? home : "/tmp") / ".mugen" / "logs";
    }

    static auto level_str(Level level) -> const char* {
        switch (level) {
            case Level::Debug: return "DEBUG";
            case Level::Info:  return "INFO";
            case Level::Warn:  return "WARN";
            case Level::Error: return "ERROR";
        }
        return "???";
    }

    static auto now_str() -> std::string {
        auto now = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
                      .count()
                  % 1000;

        struct tm tm_buf {};
        localtime_r(&tt, &tm_buf);

        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);

        char result[40];
        std::snprintf(result, sizeof(result), "%s.%03d", ts,
                      static_cast<int>(ms));
        return {result};
    }

    static auto date_str() -> std::string {
        auto tt = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        struct tm tm_buf {};
        localtime_r(&tt, &tm_buf);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y%m%d", &tm_buf);
        return {buf};
    }

    void ensure_file() {
        auto today = date_str();
        if (today == current_date && current_file.is_open()) return;

        std::error_code ec;
        std::filesystem::create_directories(log_dir, ec);

        if (current_file.is_open()) current_file.close();

        auto path = log_dir / ("mugen-" + today + ".log");
        current_file.open(path, std::ios::app);
        current_date = today;
    }

    void write(Level level, const std::string& component,
               const std::string& message) {
        std::lock_guard lock(mutex);

        if (level < min_level) return;

        ensure_file();

        if (current_file.is_open()) {
            current_file << '[' << now_str() << "] [" << level_str(level)
                         << "] [" << component << "] " << message << '\n';
            current_file.flush();
        }
    }
};

// ---------------------------------------------------------------------------
// Logger public API
// ---------------------------------------------------------------------------

Logger::Logger() : impl_(std::make_unique<Impl>()) {}
Logger::~Logger() = default;

auto Logger::instance() -> Logger& {
    static Logger logger;
    return logger;
}

void Logger::set_log_dir(const std::filesystem::path& dir) {
    std::lock_guard lock(impl_->mutex);
    impl_->log_dir = dir;
    impl_->current_date.clear();
}

void Logger::set_level(Level level) {
    std::lock_guard lock(impl_->mutex);
    impl_->min_level = level;
}

void Logger::log(Level level, const std::string& component,
                 const std::string& message) {
    impl_->write(level, component, message);
}

void Logger::debug(const std::string& component, const std::string& msg) {
    log(Level::Debug, component, msg);
}

void Logger::info(const std::string& component, const std::string& msg) {
    log(Level::Info, component, msg);
}

void Logger::warn(const std::string& component, const std::string& msg) {
    log(Level::Warn, component, msg);
}

void Logger::error(const std::string& component, const std::string& msg) {
    log(Level::Error, component, msg);
}

void Logger::flush() {
    std::lock_guard lock(impl_->mutex);
    if (impl_->current_file.is_open())
        impl_->current_file.flush();
}

}  // namespace mugen
