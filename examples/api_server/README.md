# API Server

Start a local HTTP server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint, then query it from Python, cURL, or any OpenAI SDK client.

## 1. Start the Server

```console
$ mugen-cli serve \
    --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080
```

```console
[mugen] Loading llama-2-7b-chat.Q4_K_M.gguf … done (1.2 s)
[mugen] Metal backend: Apple M3 Pro, 18 GB unified memory
[mugen] HTTP server listening on http://127.0.0.1:8080
[mugen] Endpoints:
         POST /v1/chat/completions
         POST /v1/completions
         GET  /v1/models
         GET  /health
```

## 2. cURL Examples

### Non-streaming request

```console
$ curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama-2-7b-chat",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Apple Silicon?"}
      ],
      "temperature": 0.7,
      "max_tokens": 256
    }'
```

```json
{
  "id": "mugen-a1b2c3d4",
  "object": "chat.completion",
  "created": 1717000000,
  "model": "llama-2-7b-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Apple Silicon is Apple's family of ARM-based processors designed in-house for Mac, iPad, and other devices. The M-series chips use a unified memory architecture that allows the CPU, GPU, and Neural Engine to share a single pool of high-bandwidth memory, significantly improving performance and power efficiency."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 58,
    "total_tokens": 83
  }
}
```

### Streaming request

```console
$ curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -N \
    -d '{
      "model": "llama-2-7b-chat",
      "messages": [
        {"role": "user", "content": "Count from 1 to 5."}
      ],
      "stream": true
    }'
```

```console
data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"1"},"finish_reason":null}]}

data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", 2"},"finish_reason":null}]}

data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", 3"},"finish_reason":null}]}

data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", 4"},"finish_reason":null}]}

data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", 5."},"finish_reason":null}]}

data: {"id":"mugen-e5f6g7h8","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 3. Python Client

Install the dependency:

```console
$ pip install openai
```

Run the included client script:

```console
$ python chat.py
```

See [`chat.py`](chat.py) for the full source. It demonstrates both non-streaming and streaming completions.

## 4. Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Listen port |
| `--ctx-size` | `4096` | Context window per request |
| `--max-concurrent` | `4` | Maximum concurrent requests |
| `--api-key` | *(none)* | Require Bearer token authentication |
| `--draft-model` | *(none)* | Enable speculative decoding for all requests |

### Authentication

```console
$ mugen-cli serve --model ./models/llama-2-7b-chat.Q4_K_M.gguf --api-key sk-my-secret
```

Clients must then include `Authorization: Bearer sk-my-secret` in every request, matching the standard OpenAI SDK behavior.

## 5. Tips

- **Health checks.** `GET /health` returns `{"status": "ok"}` — useful for load balancer probes.
- **Model listing.** `GET /v1/models` returns the loaded model in OpenAI-compatible format.
- **Concurrency.** Each concurrent request allocates its own KV cache. Monitor memory pressure with `--verbose` and reduce `--max-concurrent` if needed.
