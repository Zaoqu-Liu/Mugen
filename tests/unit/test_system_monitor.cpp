#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "core/monitor/system_monitor.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ===========================================================================
// SystemMonitor tests
// ===========================================================================

static void test_creation_and_state() {
    mugen::SystemMonitor mon;
    auto s = mon.current_state();

    CHECK(s.total_memory > 0);
    CHECK(s.available_memory > 0);
    CHECK(s.available_memory <= s.total_memory);
    CHECK(s.mugen_used_memory > 0);
    CHECK(s.memory_pressure >= 0.0f);
    CHECK(s.memory_pressure <= 1.0f);
    CHECK(s.ssd_available_bytes > 0);
    CHECK(s.timestamp > 0);

    std::printf("  creation_and_state PASS  (total=%zuMB, avail=%zuMB, "
                "pressure=%.2f)\n",
                s.total_memory >> 20, s.available_memory >> 20,
                s.memory_pressure);
}

static void test_degradation_level_normal() {
    mugen::SystemMonitor::Config cfg;
    cfg.warning_threshold = 0.99f;
    cfg.critical_threshold = 0.995f;
    cfg.emergency_threshold = 0.999f;
    mugen::SystemMonitor mon(cfg);

    CHECK(mon.degradation_level() == mugen::DegradationLevel::Normal ||
          mon.degradation_level() == mugen::DegradationLevel::Warning);

    std::printf("  degradation_level_normal PASS\n");
}

static void test_force_degradation() {
    mugen::SystemMonitor mon;
    CHECK(mon.degradation_level() != mugen::DegradationLevel::Emergency);

    mon.force_degradation(mugen::DegradationLevel::Emergency);
    CHECK(mon.degradation_level() == mugen::DegradationLevel::Emergency);

    mon.force_degradation(mugen::DegradationLevel::Normal);
    CHECK(mon.degradation_level() == mugen::DegradationLevel::Normal);

    std::printf("  force_degradation PASS\n");
}

static void test_degradation_callback() {
    mugen::SystemMonitor mon;

    std::atomic<int> callback_count{0};
    mugen::DegradationLevel last_old{};
    mugen::DegradationLevel last_new{};

    mon.on_degradation_change(
        [&](mugen::DegradationLevel old_lvl, mugen::DegradationLevel new_lvl) {
            last_old = old_lvl;
            last_new = new_lvl;
            callback_count.fetch_add(1);
        });

    mon.force_degradation(mugen::DegradationLevel::Critical);
    CHECK(callback_count.load() >= 1);
    CHECK(last_new == mugen::DegradationLevel::Critical);

    mon.force_degradation(mugen::DegradationLevel::Warning);
    CHECK(callback_count.load() >= 2);
    CHECK(last_old == mugen::DegradationLevel::Critical);
    CHECK(last_new == mugen::DegradationLevel::Warning);

    std::printf("  degradation_callback PASS  (fired %d times)\n",
                callback_count.load());
}

static void test_force_same_level_no_callback() {
    mugen::SystemMonitor mon;

    std::atomic<int> count{0};
    mon.on_degradation_change(
        [&](mugen::DegradationLevel, mugen::DegradationLevel) {
            count.fetch_add(1);
        });

    auto current = mon.degradation_level();
    mon.force_degradation(current);
    CHECK(count.load() == 0);

    std::printf("  force_same_level_no_callback PASS\n");
}

static void test_ssd_space_check() {
    mugen::SystemMonitor mon;

    CHECK(mon.check_ssd_space(1));
    CHECK(!mon.check_ssd_space(SIZE_MAX));

    std::printf("  ssd_space_check PASS\n");
}

static void test_format_metrics() {
    mugen::SystemMonitor mon;

    auto str = mon.format_metrics(8.3f, 0.942f, 3);

    CHECK(str.find("[Mugen]") != std::string::npos);
    CHECK(str.find("8.3 tok/s") != std::string::npos);
    CHECK(str.find("cache: 94.2%") != std::string::npos);
    CHECK(str.find("prefetch: 3") != std::string::npos);
    CHECK(str.find("mode:") != std::string::npos);
    CHECK(str.find("GB") != std::string::npos);

    std::printf("  format_metrics PASS\n  → %s\n", str.c_str());
}

static void test_start_stop() {
    mugen::SystemMonitor mon;
    mon.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mon.stop();

    auto s = mon.current_state();
    CHECK(s.total_memory > 0);

    std::printf("  start_stop PASS\n");
}

static void test_double_start_stop() {
    mugen::SystemMonitor mon;
    mon.start();
    mon.start();
    mon.stop();
    mon.stop();
    std::printf("  double_start_stop PASS\n");
}

static void test_background_updates_state() {
    mugen::SystemMonitor::Config cfg;
    cfg.poll_interval_ms = 50;
    mugen::SystemMonitor mon(cfg);

    auto t0 = mon.current_state().timestamp;
    mon.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto t1 = mon.current_state().timestamp;
    mon.stop();

    CHECK(t1 > t0);
    std::printf("  background_updates_state PASS  (dt=%llu ms)\n",
                static_cast<unsigned long long>(t1 - t0));
}

// ===========================================================================
// Logger tests
// ===========================================================================

static void test_logger_writes_file() {
    auto tmp = std::filesystem::temp_directory_path() / "mugen_test_logs";
    std::filesystem::remove_all(tmp);

    auto& logger = mugen::Logger::instance();
    logger.set_log_dir(tmp);
    logger.set_level(mugen::Logger::Level::Debug);

    logger.info("scheduler", "Pipeline swap completed");
    logger.warn("monitor", "Memory pressure rising");
    logger.debug("cache", "Hit rate 94.2%");
    logger.error("io", "SSD read timeout");
    logger.flush();

    bool found_file = false;
    for (auto& entry : std::filesystem::directory_iterator(tmp)) {
        if (entry.path().extension() == ".log") {
            found_file = true;
            std::ifstream in(entry.path());
            std::string contents((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());

            CHECK(contents.find("[INFO]") != std::string::npos);
            CHECK(contents.find("[scheduler]") != std::string::npos);
            CHECK(contents.find("Pipeline swap completed") != std::string::npos);
            CHECK(contents.find("[WARN]") != std::string::npos);
            CHECK(contents.find("[DEBUG]") != std::string::npos);
            CHECK(contents.find("[ERROR]") != std::string::npos);
        }
    }
    CHECK(found_file);

    std::filesystem::remove_all(tmp);
    std::printf("  logger_writes_file PASS\n");
}

static void test_logger_level_filter() {
    auto tmp = std::filesystem::temp_directory_path() / "mugen_test_logs_lvl";
    std::filesystem::remove_all(tmp);

    auto& logger = mugen::Logger::instance();
    logger.set_log_dir(tmp);
    logger.set_level(mugen::Logger::Level::Warn);

    logger.debug("test", "should be filtered");
    logger.info("test", "should be filtered");
    logger.warn("test", "should appear");
    logger.error("test", "should appear");
    logger.flush();

    for (auto& entry : std::filesystem::directory_iterator(tmp)) {
        if (entry.path().extension() == ".log") {
            std::ifstream in(entry.path());
            std::string contents((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());

            CHECK(contents.find("should be filtered") == std::string::npos);
            CHECK(contents.find("should appear") != std::string::npos);
        }
    }

    std::filesystem::remove_all(tmp);
    std::printf("  logger_level_filter PASS\n");
}

static void test_logger_format() {
    auto tmp = std::filesystem::temp_directory_path() / "mugen_test_logs_fmt";
    std::filesystem::remove_all(tmp);

    auto& logger = mugen::Logger::instance();
    logger.set_log_dir(tmp);
    logger.set_level(mugen::Logger::Level::Info);

    logger.info("scheduler", "Pipeline swap completed");
    logger.flush();

    for (auto& entry : std::filesystem::directory_iterator(tmp)) {
        if (entry.path().extension() == ".log") {
            std::ifstream in(entry.path());
            std::string line;
            std::getline(in, line);

            // Expected: [2026-03-23 12:34:56.789] [INFO] [scheduler] Pipeline swap completed
            CHECK(line.size() > 0);
            CHECK(line[0] == '[');
            CHECK(line.find("] [INFO] [scheduler] Pipeline swap completed")
                  != std::string::npos);
        }
    }

    std::filesystem::remove_all(tmp);
    std::printf("  logger_format PASS\n");
}

static void test_logger_thread_safety() {
    auto tmp = std::filesystem::temp_directory_path() / "mugen_test_logs_mt";
    std::filesystem::remove_all(tmp);

    auto& logger = mugen::Logger::instance();
    logger.set_log_dir(tmp);
    logger.set_level(mugen::Logger::Level::Info);

    constexpr int kThreads = 4;
    constexpr int kMessages = 50;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kMessages; ++i) {
                logger.info("thread-" + std::to_string(t),
                            "msg-" + std::to_string(i));
            }
        });
    }

    for (auto& th : threads) th.join();
    logger.flush();

    int line_count = 0;
    for (auto& entry : std::filesystem::directory_iterator(tmp)) {
        if (entry.path().extension() == ".log") {
            std::ifstream in(entry.path());
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty()) ++line_count;
            }
        }
    }

    CHECK(line_count == kThreads * kMessages);

    std::filesystem::remove_all(tmp);
    std::printf("  logger_thread_safety PASS  (%d lines from %d threads)\n",
                line_count, kThreads);
}

// ===========================================================================
int main() {
    std::printf("=== SystemMonitor state tests ===\n");
    test_creation_and_state();
    test_ssd_space_check();
    test_format_metrics();

    std::printf("=== SystemMonitor degradation tests ===\n");
    test_degradation_level_normal();
    test_force_degradation();
    test_degradation_callback();
    test_force_same_level_no_callback();

    std::printf("=== SystemMonitor background tests ===\n");
    test_start_stop();
    test_double_start_stop();
    test_background_updates_state();

    std::printf("=== Logger tests ===\n");
    test_logger_writes_file();
    test_logger_level_filter();
    test_logger_format();
    test_logger_thread_safety();

    std::printf("\nAll system monitor tests passed.\n");
    return 0;
}
