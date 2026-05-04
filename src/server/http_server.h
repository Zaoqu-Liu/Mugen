#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace mugen {

class HttpServer {
public:
    struct Config {
        uint16_t port = 8080;
        uint32_t max_connections = 10;
        std::string host = "127.0.0.1";
    };

    struct Request {
        std::string method;
        std::string path;
        std::string body;
        std::unordered_map<std::string, std::string> headers;
        std::string query_string;
    };

    struct Response {
        int status = 200;
        std::string body;
        std::string content_type = "application/json";
        std::unordered_map<std::string, std::string> headers;
    };

    using Handler = std::function<Response(const Request&)>;
    using StreamHandler = std::function<void(const Request&, std::function<void(const std::string&)>)>;

    explicit HttpServer(Config config);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void route(const std::string& method, const std::string& path, Handler handler);
    void route_stream(const std::string& method, const std::string& path, StreamHandler handler);

    auto start() -> std::expected<void, std::string>;
    void stop();

    auto is_running() const -> bool;

    // For unit-testing: parse a raw HTTP request buffer into a Request struct.
    static auto parse_request(const std::string& raw) -> std::expected<Request, std::string>;

    // For unit-testing: format a Response into a raw HTTP response string.
    static auto format_response(const Response& resp) -> std::string;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mugen
