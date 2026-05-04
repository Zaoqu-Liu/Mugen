#pragma once

#include <cstdint>
#include <string_view>

// HTTP API server exposing an OpenAI-compatible /v1/chat/completions endpoint.
// Runs a single-threaded event loop on the main thread with async I/O,
// dispatching inference requests to the engine pipeline.

namespace mugen {

class Server {
public:
    Server() = default;
    ~Server() = default;

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) noexcept = default;
    Server& operator=(Server&&) noexcept = default;
};

}  // namespace mugen
