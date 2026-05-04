#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <variant>
#include <optional>

namespace mugen::json {

// --- String escaping ---

auto escape(std::string_view raw) -> std::string;

// --- JSON builder (write-side) ---

class Value;
using Object = std::vector<std::pair<std::string, Value>>;
using Array  = std::vector<Value>;

class Value {
public:
    Value() : repr_("null") {}
    Value(std::nullptr_t) : repr_("null") {}
    Value(bool b) : repr_(b ? "true" : "false") {}
    Value(int v);
    Value(int64_t v);
    Value(uint64_t v);
    Value(double v);
    Value(const char* s);
    Value(std::string_view s);
    Value(const std::string& s);
    Value(const Object& obj);
    Value(const Array& arr);

    auto dump() const -> const std::string& { return repr_; }

private:
    std::string repr_;
};

inline auto object(std::initializer_list<std::pair<std::string, Value>> fields) -> Value {
    Object obj;
    obj.reserve(fields.size());
    for (auto& f : fields) obj.push_back(f);
    return Value(obj);
}

inline auto array(std::initializer_list<Value> items) -> Value {
    Array arr;
    arr.reserve(items.size());
    for (auto& v : items) arr.push_back(v);
    return Value(arr);
}

// --- Minimal JSON parser (read-side, for chat completions request) ---

enum class TokenKind {
    LBrace, RBrace, LBracket, RBracket,
    Colon, Comma,
    String, Number, True, False, Null,
    Eof, Error
};

struct Token {
    TokenKind kind = TokenKind::Error;
    std::string value;
};

class Parser {
public:
    explicit Parser(std::string_view input);

    struct JsonValue {
        enum Type { Null, Bool, Number, String, Array, Object } type = Null;
        std::string str_val;
        double num_val = 0.0;
        bool bool_val = false;
        std::vector<JsonValue> arr_val;
        std::vector<std::pair<std::string, JsonValue>> obj_val;

        auto get(std::string_view key) const -> const JsonValue*;
        auto as_string(std::string_view def = "") const -> std::string_view;
        auto as_bool(bool def = false) const -> bool;
        auto as_int(int def = 0) const -> int;
        auto as_double(double def = 0.0) const -> double;
    };

    auto parse() -> std::optional<JsonValue>;

private:
    auto next_token() -> Token;
    auto parse_value() -> std::optional<JsonValue>;
    auto parse_object() -> std::optional<JsonValue>;
    auto parse_array() -> std::optional<JsonValue>;
    auto parse_string_token() -> Token;
    auto parse_number_token() -> Token;
    void skip_whitespace();

    std::string_view input_;
    size_t pos_ = 0;
};

}  // namespace mugen::json
