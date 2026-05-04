# Integration Tests

End-to-end tests for the Mugen inference server. These tests start a real
`mugen-server` process and verify HTTP endpoints over the network.

## Prerequisites

1. Build the project first:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build -j$(sysctl -n hw.ncpu)
   ```

2. Python 3.8+ (uses only the standard library — no pip installs needed).

## Running

From the project root:

```bash
# Basic run (server binary auto-detected from build/):
python -m unittest discover -s tests/integration -p "test_*.py" -v

# Or directly:
python tests/integration/test_server_e2e.py -v

# With a custom server binary location:
MUGEN_SERVER_BIN=build/src/mugen-server python -m unittest tests/integration -v

# With a model loaded (enables chat completion tests):
MUGEN_TEST_MODEL=/path/to/model.gguf python tests/integration/test_server_e2e.py -v
```

## Environment Variables

| Variable           | Description                                      |
|--------------------|--------------------------------------------------|
| `MUGEN_SERVER_BIN` | Explicit path to the `mugen-server` binary.      |
| `MUGEN_TEST_MODEL` | Path to a GGUF model file for inference tests.   |

## What's Tested

- `GET /health` — liveness check
- `GET /v1/models` — model listing
- `POST /v1/chat/completions` — inference (requires model)
- `GET /nonexistent` — 404 handling
- `OPTIONS` — CORS preflight
- Malformed JSON body — error resilience
- Server stability after test suite completes
