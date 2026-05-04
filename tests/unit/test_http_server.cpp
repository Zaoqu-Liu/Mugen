#include <cstdio>
#include <cstdlib>
#include <string>

#include "server/json_utils.h"
#include "server/http_server.h"

#define MUGEN_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__,      \
                         __LINE__);                                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define MUGEN_CHECK_EQ(a, b)                                                  \
    do {                                                                      \
        if ((a) != (b)) {                                                     \
            std::fprintf(stderr, "FAIL: %s == %s (%s:%d)\n  got: \"%s\"\n  exp: \"%s\"\n", \
                         #a, #b, __FILE__, __LINE__,                          \
                         std::string(a).c_str(), std::string(b).c_str());     \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static int tests_passed = 0;
static void pass(const char* name) {
    std::printf("  PASS: %s\n", name);
    ++tests_passed;
}

// =========================================================================
// JSON escape tests
// =========================================================================

static void test_json_escape_basic() {
    MUGEN_CHECK_EQ(mugen::json::escape("hello"), "hello");
    pass("json_escape_basic");
}

static void test_json_escape_quotes() {
    MUGEN_CHECK_EQ(mugen::json::escape("say \"hi\""), "say \\\"hi\\\"");
    pass("json_escape_quotes");
}

static void test_json_escape_backslash() {
    MUGEN_CHECK_EQ(mugen::json::escape("a\\b"), "a\\\\b");
    pass("json_escape_backslash");
}

static void test_json_escape_control_chars() {
    MUGEN_CHECK_EQ(mugen::json::escape("a\nb\tc"), "a\\nb\\tc");
    pass("json_escape_control_chars");
}

static void test_json_escape_low_ascii() {
    std::string input(1, '\x01');
    MUGEN_CHECK_EQ(mugen::json::escape(input), "\\u0001");
    pass("json_escape_low_ascii");
}

// =========================================================================
// JSON builder tests
// =========================================================================

static void test_json_value_null() {
    mugen::json::Value v(nullptr);
    MUGEN_CHECK_EQ(v.dump(), "null");
    pass("json_value_null");
}

static void test_json_value_bool() {
    mugen::json::Value t(true);
    mugen::json::Value f(false);
    MUGEN_CHECK_EQ(t.dump(), "true");
    MUGEN_CHECK_EQ(f.dump(), "false");
    pass("json_value_bool");
}

static void test_json_value_int() {
    mugen::json::Value v(42);
    MUGEN_CHECK_EQ(v.dump(), "42");
    pass("json_value_int");
}

static void test_json_value_string() {
    mugen::json::Value v("hello world");
    MUGEN_CHECK_EQ(v.dump(), "\"hello world\"");
    pass("json_value_string");
}

static void test_json_object() {
    auto obj = mugen::json::object({
        {"name", "mugen"},
        {"version", 1},
    });
    auto s = obj.dump();
    MUGEN_CHECK(s.find("\"name\":\"mugen\"") != std::string::npos);
    MUGEN_CHECK(s.find("\"version\":1") != std::string::npos);
    MUGEN_CHECK(s.front() == '{');
    MUGEN_CHECK(s.back() == '}');
    pass("json_object");
}

static void test_json_array() {
    auto arr = mugen::json::array({1, 2, 3});
    MUGEN_CHECK_EQ(arr.dump(), "[1,2,3]");
    pass("json_array");
}

static void test_json_nested() {
    auto obj = mugen::json::object({
        {"choices", mugen::json::array({
            mugen::json::object({
                {"index", 0},
                {"message", mugen::json::object({
                    {"role", "assistant"},
                    {"content", "hi"},
                })},
            })
        })},
    });
    auto s = obj.dump();
    MUGEN_CHECK(s.find("\"choices\":[") != std::string::npos);
    MUGEN_CHECK(s.find("\"role\":\"assistant\"") != std::string::npos);
    pass("json_nested");
}

// =========================================================================
// JSON parser tests
// =========================================================================

static void test_json_parse_object() {
    std::string input = R"({"model":"deepseek-v3","stream":true,"max_tokens":256,"temperature":0.7})";
    mugen::json::Parser parser(input);
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK(result->type == mugen::json::Parser::JsonValue::Object);

    auto* model = result->get("model");
    MUGEN_CHECK(model != nullptr);
    MUGEN_CHECK_EQ(std::string(model->as_string()), "deepseek-v3");

    auto* stream = result->get("stream");
    MUGEN_CHECK(stream != nullptr);
    MUGEN_CHECK(stream->as_bool() == true);

    auto* max_tokens = result->get("max_tokens");
    MUGEN_CHECK(max_tokens != nullptr);
    MUGEN_CHECK(max_tokens->as_int() == 256);

    auto* temp = result->get("temperature");
    MUGEN_CHECK(temp != nullptr);
    MUGEN_CHECK(temp->as_double() > 0.69 && temp->as_double() < 0.71);

    pass("json_parse_object");
}

static void test_json_parse_messages() {
    std::string input = R"({"messages":[{"role":"user","content":"Hello"}]})";
    mugen::json::Parser parser(input);
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());

    auto* messages = result->get("messages");
    MUGEN_CHECK(messages != nullptr);
    MUGEN_CHECK(messages->type == mugen::json::Parser::JsonValue::Array);
    MUGEN_CHECK(messages->arr_val.size() == 1);

    auto& msg = messages->arr_val[0];
    auto* role = msg.get("role");
    MUGEN_CHECK(role != nullptr);
    MUGEN_CHECK_EQ(std::string(role->as_string()), "user");

    auto* content = msg.get("content");
    MUGEN_CHECK(content != nullptr);
    MUGEN_CHECK_EQ(std::string(content->as_string()), "Hello");

    pass("json_parse_messages");
}

static void test_json_parse_empty_object() {
    mugen::json::Parser parser("{}");
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK(result->type == mugen::json::Parser::JsonValue::Object);
    MUGEN_CHECK(result->obj_val.empty());
    pass("json_parse_empty_object");
}

static void test_json_parse_null_false() {
    std::string input = R"({"a":null,"b":false})";
    mugen::json::Parser parser(input);
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());
    auto* a = result->get("a");
    MUGEN_CHECK(a != nullptr);
    MUGEN_CHECK(a->type == mugen::json::Parser::JsonValue::Null);
    auto* b = result->get("b");
    MUGEN_CHECK(b != nullptr);
    MUGEN_CHECK(b->as_bool() == false);
    pass("json_parse_null_false");
}

static void test_json_parse_escaped_string() {
    std::string input = R"({"text":"line1\nline2\ttab"})";
    mugen::json::Parser parser(input);
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());
    auto* text = result->get("text");
    MUGEN_CHECK(text != nullptr);
    MUGEN_CHECK_EQ(std::string(text->as_string()), "line1\nline2\ttab");
    pass("json_parse_escaped_string");
}

static void test_json_parse_negative_number() {
    std::string input = R"({"val":-42})";
    mugen::json::Parser parser(input);
    auto result = parser.parse();
    MUGEN_CHECK(result.has_value());
    auto* val = result->get("val");
    MUGEN_CHECK(val != nullptr);
    MUGEN_CHECK(val->as_int() == -42);
    pass("json_parse_negative_number");
}

// =========================================================================
// HTTP request parsing tests
// =========================================================================

static void test_parse_get_request() {
    std::string raw =
        "GET /v1/models HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Accept: application/json\r\n"
        "\r\n";

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK_EQ(result->method, "GET");
    MUGEN_CHECK_EQ(result->path, "/v1/models");
    MUGEN_CHECK(result->body.empty());

    auto it = result->headers.find("host");
    MUGEN_CHECK(it != result->headers.end());
    MUGEN_CHECK_EQ(it->second, "localhost:8080");

    pass("parse_get_request");
}

static void test_parse_post_request() {
    std::string body = R"({"model":"deepseek-v3","messages":[{"role":"user","content":"Hello"}]})";
    std::string raw =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" + body;

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK_EQ(result->method, "POST");
    MUGEN_CHECK_EQ(result->path, "/v1/chat/completions");
    MUGEN_CHECK_EQ(result->body, body);
    pass("parse_post_request");
}

static void test_parse_query_string() {
    std::string raw =
        "GET /v1/models?format=json HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK_EQ(result->path, "/v1/models");
    MUGEN_CHECK_EQ(result->query_string, "format=json");
    pass("parse_query_string");
}

static void test_parse_chunked_request() {
    std::string raw =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
        "5\r\n"
        "Hello\r\n"
        "7\r\n"
        " World!\r\n"
        "0\r\n"
        "\r\n";

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK_EQ(result->body, "Hello World!");
    pass("parse_chunked_request");
}

static void test_parse_incomplete_request() {
    std::string raw = "GET /v1/models HTTP/1.1\r\nHost: localhost";
    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(!result.has_value());
    pass("parse_incomplete_request");
}

// =========================================================================
// HTTP response formatting tests
// =========================================================================

static void test_format_response() {
    mugen::HttpServer::Response resp;
    resp.status = 200;
    resp.body = R"({"status":"ok"})";
    resp.content_type = "application/json";

    auto formatted = mugen::HttpServer::format_response(resp);
    MUGEN_CHECK(formatted.find("HTTP/1.1 200 OK") != std::string::npos);
    MUGEN_CHECK(formatted.find("Content-Type: application/json") != std::string::npos);
    MUGEN_CHECK(formatted.find("Content-Length: 15") != std::string::npos);
    MUGEN_CHECK(formatted.find("Access-Control-Allow-Origin: *") != std::string::npos);
    MUGEN_CHECK(formatted.find(R"({"status":"ok"})") != std::string::npos);
    pass("format_response");
}

static void test_format_404_response() {
    mugen::HttpServer::Response resp;
    resp.status = 404;
    resp.body = R"({"error":"Not Found"})";

    auto formatted = mugen::HttpServer::format_response(resp);
    MUGEN_CHECK(formatted.find("HTTP/1.1 404 Not Found") != std::string::npos);
    pass("format_404_response");
}

// =========================================================================
// Header parsing edge cases
// =========================================================================

static void test_parse_case_insensitive_headers() {
    std::string body = "test";
    std::string raw =
        "POST /test HTTP/1.1\r\n"
        "CONTENT-LENGTH: 4\r\n"
        "CONTENT-TYPE: text/plain\r\n"
        "\r\n"
        "test";

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK_EQ(result->body, "test");

    auto it = result->headers.find("content-type");
    MUGEN_CHECK(it != result->headers.end());
    MUGEN_CHECK_EQ(it->second, "text/plain");

    pass("parse_case_insensitive_headers");
}

static void test_parse_multiple_headers() {
    std::string raw =
        "GET /test HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Accept: application/json\r\n"
        "Authorization: Bearer token123\r\n"
        "X-Custom: value\r\n"
        "\r\n";

    auto result = mugen::HttpServer::parse_request(raw);
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK(result->headers.size() >= 4);
    MUGEN_CHECK_EQ(result->headers["authorization"], "Bearer token123");
    MUGEN_CHECK_EQ(result->headers["x-custom"], "value");
    pass("parse_multiple_headers");
}

// =========================================================================
// main
// =========================================================================

int main() {
    std::printf("=== JSON escape tests ===\n");
    test_json_escape_basic();
    test_json_escape_quotes();
    test_json_escape_backslash();
    test_json_escape_control_chars();
    test_json_escape_low_ascii();

    std::printf("=== JSON builder tests ===\n");
    test_json_value_null();
    test_json_value_bool();
    test_json_value_int();
    test_json_value_string();
    test_json_object();
    test_json_array();
    test_json_nested();

    std::printf("=== JSON parser tests ===\n");
    test_json_parse_object();
    test_json_parse_messages();
    test_json_parse_empty_object();
    test_json_parse_null_false();
    test_json_parse_escaped_string();
    test_json_parse_negative_number();

    std::printf("=== HTTP request parsing tests ===\n");
    test_parse_get_request();
    test_parse_post_request();
    test_parse_query_string();
    test_parse_chunked_request();
    test_parse_incomplete_request();

    std::printf("=== HTTP response formatting tests ===\n");
    test_format_response();
    test_format_404_response();

    std::printf("=== HTTP header edge case tests ===\n");
    test_parse_case_insensitive_headers();
    test_parse_multiple_headers();

    std::printf("\nAll %d tests passed.\n", tests_passed);
    return 0;
}
