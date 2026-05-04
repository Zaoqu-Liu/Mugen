#include <cstdint>
#include <string>

#include "server/http_server.h"

// Feed arbitrary bytes to the HTTP request parser.
// The parser must return an error or a valid Request — never crash.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::string raw(reinterpret_cast<const char*>(data), size);

    auto result = mugen::HttpServer::parse_request(raw);
    if (result.has_value()) {
        // Exercise accessors on the parsed request.
        [[maybe_unused]] auto& method = result->method;
        [[maybe_unused]] auto& path   = result->path;
        [[maybe_unused]] auto& body   = result->body;
        [[maybe_unused]] auto& qs     = result->query_string;

        for (const auto& [k, v] : result->headers) {
            [[maybe_unused]] auto len = k.size() + v.size();
        }

        // Round-trip: format a response for this parsed request.
        mugen::HttpServer::Response resp;
        resp.status = 200;
        resp.body = body;
        [[maybe_unused]] auto formatted = mugen::HttpServer::format_response(resp);
    }

    return 0;
}
