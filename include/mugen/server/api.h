#pragma once

#include <cstdint>
#include <string_view>

// Public types for the HTTP API server.

namespace mugen {

struct ServerConfig {
    std::string_view host = "127.0.0.1";
    uint16_t port         = 8080;
    uint32_t max_connections = 64;
};

}  // namespace mugen
