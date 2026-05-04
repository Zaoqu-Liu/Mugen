#include "server/http_server.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace mugen {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto status_text(int code) -> const char* {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        default:  return "Unknown";
    }
}

static auto to_lower(std::string s) -> std::string {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static constexpr size_t kMaxRequestSize = 4 * 1024 * 1024; // 4 MB

// ---------------------------------------------------------------------------
// Request parsing (static, also used by tests)
// ---------------------------------------------------------------------------

auto HttpServer::parse_request(const std::string& raw) -> std::expected<Request, std::string> {
    auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return std::unexpected("Incomplete HTTP request: no header terminator");
    }

    auto first_line_end = raw.find("\r\n");
    if (first_line_end == std::string::npos) {
        return std::unexpected("Malformed request line");
    }

    std::string request_line = raw.substr(0, first_line_end);

    // Parse "METHOD PATH HTTP/1.x"
    auto sp1 = request_line.find(' ');
    if (sp1 == std::string::npos)
        return std::unexpected("Malformed request line: no method");

    auto sp2 = request_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos)
        return std::unexpected("Malformed request line: no path");

    Request req;
    req.method = request_line.substr(0, sp1);
    std::string full_path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    auto qmark = full_path.find('?');
    if (qmark != std::string::npos) {
        req.path = full_path.substr(0, qmark);
        req.query_string = full_path.substr(qmark + 1);
    } else {
        req.path = full_path;
    }

    // Parse headers
    size_t pos = first_line_end + 2;
    while (pos < header_end) {
        auto line_end = raw.find("\r\n", pos);
        if (line_end == std::string::npos || line_end > header_end) break;

        std::string line = raw.substr(pos, line_end - pos);
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = to_lower(line.substr(0, colon));
            std::string val = line.substr(colon + 1);
            // Trim leading whitespace from value
            size_t vstart = val.find_first_not_of(" \t");
            if (vstart != std::string::npos) val = val.substr(vstart);
            req.headers[key] = val;
        }
        pos = line_end + 2;
    }

    // Extract body
    size_t body_start = header_end + 4;

    auto cl_it = req.headers.find("content-length");
    if (cl_it != req.headers.end()) {
        size_t content_len = 0;
        try { content_len = std::stoull(cl_it->second); } catch (...) { content_len = 0; }
        if (body_start + content_len <= raw.size()) {
            req.body = raw.substr(body_start, content_len);
        } else {
            req.body = raw.substr(body_start);
        }
    } else {
        auto te_it = req.headers.find("transfer-encoding");
        if (te_it != req.headers.end() && te_it->second.find("chunked") != std::string::npos) {
            // Decode chunked transfer-encoding
            std::string decoded;
            size_t cpos = body_start;
            while (cpos < raw.size()) {
                auto chunk_end = raw.find("\r\n", cpos);
                if (chunk_end == std::string::npos) break;
                std::string hex_str = raw.substr(cpos, chunk_end - cpos);
                size_t chunk_size = 0;
                try { chunk_size = std::stoull(hex_str, nullptr, 16); } catch (...) { break; }
                if (chunk_size == 0) break;
                cpos = chunk_end + 2;
                if (cpos + chunk_size > raw.size()) break;
                decoded += raw.substr(cpos, chunk_size);
                cpos += chunk_size + 2; // skip chunk data + trailing CRLF
            }
            req.body = std::move(decoded);
        } else if (body_start < raw.size()) {
            req.body = raw.substr(body_start);
        }
    }

    return req;
}

// ---------------------------------------------------------------------------
// Response formatting (static)
// ---------------------------------------------------------------------------

static void append_cors_headers(std::string& out) {
    out += "Access-Control-Allow-Origin: *\r\n";
    out += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    out += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
}

auto HttpServer::format_response(const Response& resp) -> std::string {
    std::string out;
    out += "HTTP/1.1 ";
    out += std::to_string(resp.status);
    out += " ";
    out += status_text(resp.status);
    out += "\r\n";

    out += "Content-Type: " + resp.content_type + "\r\n";
    out += "Content-Length: " + std::to_string(resp.body.size()) + "\r\n";
    out += "Connection: close\r\n";
    append_cors_headers(out);

    for (auto& [k, v] : resp.headers) {
        out += k + ": " + v + "\r\n";
    }

    out += "\r\n";
    out += resp.body;
    return out;
}

// ---------------------------------------------------------------------------
// Impl (pimpl)
// ---------------------------------------------------------------------------

struct RouteKey {
    std::string method;
    std::string path;

    bool operator==(const RouteKey& other) const {
        return method == other.method && path == other.path;
    }
};

struct RouteKeyHash {
    size_t operator()(const RouteKey& k) const {
        auto h1 = std::hash<std::string>{}(k.method);
        auto h2 = std::hash<std::string>{}(k.path);
        return h1 ^ (h2 << 1);
    }
};

struct HttpServer::Impl {
    Config config;
    int listen_fd = -1;
    std::atomic<bool> running{false};
    std::thread accept_thread;
    std::mutex threads_mutex;
    std::vector<std::thread> worker_threads;

    std::unordered_map<RouteKey, Handler, RouteKeyHash> handlers;
    std::unordered_map<RouteKey, StreamHandler, RouteKeyHash> stream_handlers;

    void accept_loop();
    void handle_connection(int client_fd);

    auto read_full_request(int fd) -> std::expected<std::string, std::string>;

    void send_all(int fd, const std::string& data);
    void send_stream_response(int fd, const Request& req, const StreamHandler& handler);
    void send_options_response(int fd, const Request& req);
};

// Fully read an HTTP request (headers + body) from the socket.
auto HttpServer::Impl::read_full_request(int fd) -> std::expected<std::string, std::string> {
    std::string buf;
    buf.reserve(4096);
    char tmp[4096];

    // Read until we have the full headers
    while (buf.find("\r\n\r\n") == std::string::npos) {
        auto n = ::read(fd, tmp, sizeof(tmp));
        if (n <= 0) return std::unexpected("Connection closed or read error");
        buf.append(tmp, static_cast<size_t>(n));
        if (buf.size() > kMaxRequestSize) return std::unexpected("Request too large");
    }

    auto header_end = buf.find("\r\n\r\n");
    size_t body_start = header_end + 4;

    // Check Content-Length to know how much body to read
    auto cl_pos = buf.find("Content-Length:");
    if (cl_pos == std::string::npos) cl_pos = buf.find("content-length:");
    if (cl_pos != std::string::npos && cl_pos < header_end) {
        auto val_start = cl_pos + 15; // strlen("Content-Length:")
        while (val_start < header_end && (buf[val_start] == ' ' || buf[val_start] == '\t')) ++val_start;
        auto val_end = buf.find("\r\n", val_start);
        if (val_end == std::string::npos) val_end = header_end;
        size_t content_len = 0;
        try { content_len = std::stoull(buf.substr(val_start, val_end - val_start)); } catch (...) {}

        size_t needed = body_start + content_len;
        while (buf.size() < needed) {
            auto n = ::read(fd, tmp, sizeof(tmp));
            if (n <= 0) break;
            buf.append(tmp, static_cast<size_t>(n));
            if (buf.size() > kMaxRequestSize) return std::unexpected("Request too large");
        }
    }
    // For chunked, we read until the terminating 0-size chunk
    auto te_pos = buf.find("Transfer-Encoding:");
    if (te_pos == std::string::npos) te_pos = buf.find("transfer-encoding:");
    if (te_pos != std::string::npos && te_pos < header_end) {
        // Keep reading until we see "0\r\n\r\n"
        while (buf.find("0\r\n\r\n", body_start) == std::string::npos) {
            auto n = ::read(fd, tmp, sizeof(tmp));
            if (n <= 0) break;
            buf.append(tmp, static_cast<size_t>(n));
            if (buf.size() > kMaxRequestSize) return std::unexpected("Request too large");
        }
    }

    return buf;
}

void HttpServer::Impl::send_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        auto n = ::write(fd, data.data() + sent, data.size() - sent);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

void HttpServer::Impl::send_options_response(int fd, [[maybe_unused]] const Request& req) {
    std::string out;
    out += "HTTP/1.1 204 No Content\r\n";
    append_cors_headers(out);
    out += "Content-Length: 0\r\n";
    out += "Connection: close\r\n";
    out += "\r\n";
    send_all(fd, out);
}

void HttpServer::Impl::send_stream_response(int fd, const Request& req, const StreamHandler& handler) {
    // The stream handler gets full control: it sends its own headers + body.
    handler(req, [this, fd](const std::string& chunk) {
        send_all(fd, chunk);
    });
}

void HttpServer::Impl::handle_connection(int client_fd) {
    auto raw_result = read_full_request(client_fd);
    if (!raw_result) {
        ::close(client_fd);
        return;
    }

    auto req_result = HttpServer::parse_request(*raw_result);
    if (!req_result) {
        Response err_resp;
        err_resp.status = 400;
        err_resp.body = R"({"error":"Bad Request"})";
        send_all(client_fd, HttpServer::format_response(err_resp));
        ::close(client_fd);
        return;
    }

    auto& req = *req_result;

    // Handle CORS preflight
    if (req.method == "OPTIONS") {
        send_options_response(client_fd, req);
        ::close(client_fd);
        return;
    }

    // Check stream handlers first
    RouteKey key{req.method, req.path};
    auto sit = stream_handlers.find(key);
    if (sit != stream_handlers.end()) {
        send_stream_response(client_fd, req, sit->second);
        ::close(client_fd);
        return;
    }

    // Regular handlers
    auto hit = handlers.find(key);
    if (hit != handlers.end()) {
        auto resp = hit->second(req);
        send_all(client_fd, HttpServer::format_response(resp));
        ::close(client_fd);
        return;
    }

    // 404
    Response not_found;
    not_found.status = 404;
    not_found.body = R"({"error":"Not Found"})";
    send_all(client_fd, HttpServer::format_response(not_found));
    ::close(client_fd);
}

void HttpServer::Impl::accept_loop() {
    while (running.load(std::memory_order_relaxed)) {
        struct sockaddr_in client_addr {};
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = ::accept(listen_fd,
                                  reinterpret_cast<struct sockaddr*>(&client_addr),
                                  &addr_len);
        if (client_fd < 0) {
            if (!running.load(std::memory_order_relaxed)) break;
            continue;
        }

        std::lock_guard lock(threads_mutex);
        // Clean up finished threads
        std::erase_if(worker_threads, [](std::thread& t) {
            if (t.joinable()) {
                // We can't cheaply check if done; just try a non-blocking approach.
                // For simplicity, detach old threads beyond max_connections.
                return false;
            }
            return true;
        });

        worker_threads.emplace_back([this, client_fd]() {
            handle_connection(client_fd);
        });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

HttpServer::HttpServer(Config config) : impl_(std::make_unique<Impl>()) {
    impl_->config = std::move(config);
}

HttpServer::~HttpServer() {
    stop();
    // Join all worker threads
    std::lock_guard lock(impl_->threads_mutex);
    for (auto& t : impl_->worker_threads) {
        if (t.joinable()) t.join();
    }
}

void HttpServer::route(const std::string& method, const std::string& path, Handler handler) {
    impl_->handlers[RouteKey{method, path}] = std::move(handler);
}

void HttpServer::route_stream(const std::string& method, const std::string& path, StreamHandler handler) {
    impl_->stream_handlers[RouteKey{method, path}] = std::move(handler);
}

auto HttpServer::start() -> std::expected<void, std::string> {
    impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->listen_fd < 0) {
        return std::unexpected(std::string("socket() failed: ") + std::strerror(errno));
    }

    int opt = 1;
    ::setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(impl_->config.port);

    if (::inet_pton(AF_INET, impl_->config.host.c_str(), &addr.sin_addr) <= 0) {
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
        return std::unexpected("Invalid host address: " + impl_->config.host);
    }

    if (::bind(impl_->listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        auto err = std::string("bind() failed: ") + std::strerror(errno);
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
        return std::unexpected(err);
    }

    if (::listen(impl_->listen_fd, static_cast<int>(impl_->config.max_connections)) < 0) {
        auto err = std::string("listen() failed: ") + std::strerror(errno);
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
        return std::unexpected(err);
    }

    impl_->running.store(true, std::memory_order_release);
    impl_->accept_thread = std::thread([this]() { impl_->accept_loop(); });

    return {};
}

void HttpServer::stop() {
    if (!impl_->running.exchange(false, std::memory_order_acq_rel)) return;

    if (impl_->listen_fd >= 0) {
        ::shutdown(impl_->listen_fd, SHUT_RDWR);
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
    }

    if (impl_->accept_thread.joinable()) {
        impl_->accept_thread.join();
    }
}

auto HttpServer::is_running() const -> bool {
    return impl_->running.load(std::memory_order_acquire);
}

}  // namespace mugen
