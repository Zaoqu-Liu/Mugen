#include "server/json_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mugen::json {

// ---------------------------------------------------------------------------
// String escaping
// ---------------------------------------------------------------------------

auto escape(std::string_view raw) -> std::string {
    std::string out;
    out.reserve(raw.size() + 8);
    for (char c : raw) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Value constructors (write-side JSON builder)
// ---------------------------------------------------------------------------

Value::Value(int v) : repr_(std::to_string(v)) {}

Value::Value(int64_t v) : repr_(std::to_string(v)) {}

Value::Value(uint64_t v) : repr_(std::to_string(v)) {}

Value::Value(double v) {
    if (std::isnan(v) || std::isinf(v)) {
        repr_ = "null";
    } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        repr_ = buf;
    }
}

Value::Value(const char* s) {
    repr_ = "\"" + escape(s) + "\"";
}

Value::Value(std::string_view s) {
    repr_ = "\"" + escape(s) + "\"";
}

Value::Value(const std::string& s) {
    repr_ = "\"" + escape(s) + "\"";
}

Value::Value(const Object& obj) {
    repr_ = "{";
    for (size_t i = 0; i < obj.size(); ++i) {
        if (i > 0) repr_ += ",";
        repr_ += "\"" + escape(obj[i].first) + "\":" + obj[i].second.dump();
    }
    repr_ += "}";
}

Value::Value(const Array& arr) {
    repr_ = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) repr_ += ",";
        repr_ += arr[i].dump();
    }
    repr_ += "]";
}

// ---------------------------------------------------------------------------
// Parser implementation
// ---------------------------------------------------------------------------

Parser::Parser(std::string_view input) : input_(input) {}

void Parser::skip_whitespace() {
    while (pos_ < input_.size() &&
           (input_[pos_] == ' '  || input_[pos_] == '\t' ||
            input_[pos_] == '\n' || input_[pos_] == '\r')) {
        ++pos_;
    }
}

auto Parser::parse_string_token() -> Token {
    // pos_ is on the opening '"'
    ++pos_;
    std::string val;
    while (pos_ < input_.size()) {
        char c = input_[pos_++];
        if (c == '"') return {TokenKind::String, std::move(val)};
        if (c == '\\' && pos_ < input_.size()) {
            char esc = input_[pos_++];
            switch (esc) {
                case '"':  val += '"';  break;
                case '\\': val += '\\'; break;
                case '/':  val += '/';  break;
                case 'b':  val += '\b'; break;
                case 'f':  val += '\f'; break;
                case 'n':  val += '\n'; break;
                case 'r':  val += '\r'; break;
                case 't':  val += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > input_.size()) return {TokenKind::Error, {}};
                    auto hex = input_.substr(pos_, 4);
                    pos_ += 4;
                    unsigned cp = 0;
                    for (char h : hex) {
                        cp <<= 4;
                        if (h >= '0' && h <= '9')      cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else return {TokenKind::Error, {}};
                    }
                    if (cp < 0x80) {
                        val += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        val += static_cast<char>(0xC0 | (cp >> 6));
                        val += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        val += static_cast<char>(0xE0 | (cp >> 12));
                        val += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        val += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: val += esc;
            }
        } else {
            val += c;
        }
    }
    return {TokenKind::Error, {}};
}

auto Parser::parse_number_token() -> Token {
    size_t start = pos_;
    if (pos_ < input_.size() && input_[pos_] == '-') ++pos_;
    while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
    if (pos_ < input_.size() && input_[pos_] == '.') {
        ++pos_;
        while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
    }
    if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
        ++pos_;
        if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
        while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
    }
    return {TokenKind::Number, std::string(input_.substr(start, pos_ - start))};
}

auto Parser::next_token() -> Token {
    skip_whitespace();
    if (pos_ >= input_.size()) return {TokenKind::Eof, {}};

    char c = input_[pos_];
    switch (c) {
        case '{': ++pos_; return {TokenKind::LBrace, "{"};
        case '}': ++pos_; return {TokenKind::RBrace, "}"};
        case '[': ++pos_; return {TokenKind::LBracket, "["};
        case ']': ++pos_; return {TokenKind::RBracket, "]"};
        case ':': ++pos_; return {TokenKind::Colon, ":"};
        case ',': ++pos_; return {TokenKind::Comma, ","};
        case '"': return parse_string_token();
        case 't':
            if (input_.substr(pos_, 4) == "true") { pos_ += 4; return {TokenKind::True, "true"}; }
            return {TokenKind::Error, {}};
        case 'f':
            if (input_.substr(pos_, 5) == "false") { pos_ += 5; return {TokenKind::False, "false"}; }
            return {TokenKind::Error, {}};
        case 'n':
            if (input_.substr(pos_, 4) == "null") { pos_ += 4; return {TokenKind::Null, "null"}; }
            return {TokenKind::Error, {}};
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number_token();
            return {TokenKind::Error, {}};
    }
}

auto Parser::parse() -> std::optional<JsonValue> {
    return parse_value();
}

auto Parser::parse_value() -> std::optional<JsonValue> {
    auto tok = next_token();
    switch (tok.kind) {
        case TokenKind::LBrace: return parse_object();
        case TokenKind::LBracket: return parse_array();
        case TokenKind::String: {
            JsonValue v;
            v.type = JsonValue::String;
            v.str_val = std::move(tok.value);
            return v;
        }
        case TokenKind::Number: {
            JsonValue v;
            v.type = JsonValue::Number;
            double d = 0.0;
            try { d = std::stod(tok.value); } catch (...) { d = 0.0; }
            v.num_val = d;
            v.str_val = std::move(tok.value);
            return v;
        }
        case TokenKind::True: {
            JsonValue v;
            v.type = JsonValue::Bool;
            v.bool_val = true;
            return v;
        }
        case TokenKind::False: {
            JsonValue v;
            v.type = JsonValue::Bool;
            v.bool_val = false;
            return v;
        }
        case TokenKind::Null: return JsonValue{};
        default: return std::nullopt;
    }
}

auto Parser::parse_object() -> std::optional<JsonValue> {
    JsonValue result;
    result.type = JsonValue::Object;

    skip_whitespace();
    if (pos_ < input_.size() && input_[pos_] == '}') {
        ++pos_;
        return result;
    }

    while (true) {
        auto key_tok = next_token();
        if (key_tok.kind != TokenKind::String) return std::nullopt;

        auto colon = next_token();
        if (colon.kind != TokenKind::Colon) return std::nullopt;

        auto val = parse_value();
        if (!val) return std::nullopt;

        result.obj_val.emplace_back(std::move(key_tok.value), std::move(*val));

        skip_whitespace();
        if (pos_ >= input_.size()) return std::nullopt;
        if (input_[pos_] == '}') { ++pos_; return result; }
        if (input_[pos_] == ',') { ++pos_; continue; }
        return std::nullopt;
    }
}

auto Parser::parse_array() -> std::optional<JsonValue> {
    JsonValue result;
    result.type = JsonValue::Array;

    skip_whitespace();
    if (pos_ < input_.size() && input_[pos_] == ']') {
        ++pos_;
        return result;
    }

    while (true) {
        auto val = parse_value();
        if (!val) return std::nullopt;

        result.arr_val.push_back(std::move(*val));

        skip_whitespace();
        if (pos_ >= input_.size()) return std::nullopt;
        if (input_[pos_] == ']') { ++pos_; return result; }
        if (input_[pos_] == ',') { ++pos_; continue; }
        return std::nullopt;
    }
}

// --- JsonValue accessors ---

auto Parser::JsonValue::get(std::string_view key) const -> const JsonValue* {
    if (type != Object) return nullptr;
    for (auto& [k, v] : obj_val) {
        if (k == key) return &v;
    }
    return nullptr;
}

auto Parser::JsonValue::as_string(std::string_view def) const -> std::string_view {
    if (type == String) return str_val;
    return def;
}

auto Parser::JsonValue::as_bool(bool def) const -> bool {
    if (type == Bool) return bool_val;
    return def;
}

auto Parser::JsonValue::as_int(int def) const -> int {
    if (type == Number) return static_cast<int>(num_val);
    return def;
}

auto Parser::JsonValue::as_double(double def) const -> double {
    if (type == Number) return num_val;
    return def;
}

}  // namespace mugen::json
