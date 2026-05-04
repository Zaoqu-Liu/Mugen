# API Reference

> OpenAI-compatible HTTP API for the Mugen inference engine.

## Starting the Server

```sh
./build/src/mugen-cli serve <model.gguf> --port 8080
```

The server binds to `127.0.0.1` by default. Endpoints are compatible with OpenAI's Chat Completions API, allowing drop-in replacement with existing tools and libraries.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/metrics` | Runtime metrics |
| `POST` | `/v1/chat/completions` | Chat completion (stream + non-stream) |

---

## `GET /health`

Returns server health status.

### Response

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### cURL Example

```sh
curl http://localhost:8080/health
```

---

## `GET /v1/models`

Lists models loaded on the server. Compatible with the OpenAI List Models endpoint.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-7b-q4_0",
      "object": "model",
      "created": 1714838400,
      "owned_by": "mugen"
    }
  ]
}
```

### cURL Example

```sh
curl http://localhost:8080/v1/models
```

---

## `GET /v1/metrics`

Returns runtime inference metrics.

### Response

```json
{
  "tok_per_sec": 56.1,
  "cache_hit_rate": 0.95,
  "memory_pressure": 0.42,
  "mode": "inference"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tok_per_sec` | `float` | Current decode throughput |
| `cache_hit_rate` | `float` | Expert cache hit rate (0.0–1.0) |
| `memory_pressure` | `float` | System memory pressure (0.0–1.0) |
| `mode` | `string` | Current mode: `"inference"`, `"idle"` |

### cURL Example

```sh
curl http://localhost:8080/v1/metrics
```

---

## `POST /v1/chat/completions`

Generate a chat completion. Supports both streaming (SSE) and non-streaming modes.

### Request Body

```json
{
  "model": "qwen2.5-7b-q4_0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": false,
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | Server's loaded model | Model identifier (informational) |
| `messages` | `array` | *required* | Array of message objects |
| `messages[].role` | `string` | — | `"system"`, `"user"`, or `"assistant"` |
| `messages[].content` | `string` | — | Message content |
| `stream` | `boolean` | `false` | Enable Server-Sent Events streaming |
| `max_tokens` | `integer` | `512` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |

### Non-Streaming Response

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6g7h8i9j0k1l2",
  "object": "chat.completion",
  "created": 1714838400,
  "model": "qwen2.5-7b-q4_0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

### Streaming Response

When `stream: true`, the response uses Server-Sent Events (SSE):

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1714838400,"model":"qwen2.5-7b-q4_0","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1714838400,"model":"qwen2.5-7b-q4_0","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1714838400,"model":"qwen2.5-7b-q4_0","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1714838400,"model":"qwen2.5-7b-q4_0","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Finish Reasons

| Value | Description |
|-------|-------------|
| `"stop"` | Natural stop (EOS token or chat template end token) |
| `"length"` | Reached `max_tokens` limit |

### cURL Examples

**Non-streaming:**

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-q4_0",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Streaming:**

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "qwen2.5-7b-q4_0",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "stream": true,
    "max_tokens": 50
  }'
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Mugen doesn't require auth
)

response = client.chat.completions.create(
    model="qwen2.5-7b-q4_0",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "messages array is required and must not be empty",
    "type": "invalid_request_error"
  }
}
```

### Error Codes

| HTTP Status | Condition |
|------------|-----------|
| `400 Bad Request` | Invalid JSON, missing `messages`, empty messages array |
| `500 Internal Server Error` | Prefill or generation failure |

---

## KV Cache Prefix Reuse

The server automatically reuses KV cache across requests in the same session. Multi-turn conversations benefit from this: only new tokens are processed during prefill, significantly reducing time-to-first-token.

The server tracks the longest common prefix between the current request's token sequence and the previous one. If a prefix match is found within the cached KV length, only the new suffix tokens are processed.

---

## CORS

All responses include permissive CORS headers:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

---

## Server Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--port N` | `8080` | Listening port |
| Host | `127.0.0.1` | Bind address (localhost only) |
| Max connections | `10` | Concurrent connection limit |
