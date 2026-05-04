# Mugen Test Suite

Three-tier test architecture: **unit → integration → fuzz**.

## Quick Start

```bash
# Build everything (tests enabled by default)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(sysctl -n hw.ncpu)

# Run unit tests
cd build && ctest --output-on-failure

# Run integration tests (needs built server)
python -m unittest discover -s tests/integration -p "test_*.py" -v

# Run fuzz tests (needs Clang)
cmake -B build-fuzz -DMUGEN_BUILD_FUZZ=ON \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz --target fuzz_gguf_parser
./build-fuzz/tests/fuzz/fuzz_gguf_parser -max_total_time=60
```

## Test Categories

### Unit Tests (`tests/unit/`)

Self-contained C++ tests using the `MUGEN_CHECK` / `CHECK` macro (zero
third-party dependencies). Each file compiles to its own executable and is
registered with CTest automatically.

| File                       | Module                 | Key areas                                         |
|----------------------------|------------------------|----------------------------------------------------|
| `test_gguf_parser.cpp`     | GGUF parser + ggml     | Valid/invalid files, MoE, arrays, boundary cases   |
| `test_http_server.cpp`     | HTTP server + JSON     | Request parsing, JSON builder/parser, edge cases   |
| `test_tokenizer.cpp`       | Tokenizer (SPM/BPE)   | Encode/decode, byte-fallback, special tokens       |
| `test_transformer.cpp`     | Transformer pipeline   | Forward pass, shapes, attention                    |
| `test_kv_cache.cpp`        | KV cache               | Insert/evict/resize                                |
| `test_buffer_manager.cpp`  | Buffer manager         | Allocation, pooling, OOM                           |
| `test_mmap_loader.cpp`     | Memory-mapped I/O      | Map/unmap, aligned access                          |
| `test_route_predictor.cpp` | Expert router          | Prediction accuracy, top-K                         |
| `test_uspp_scheduler.cpp`  | USPP scheduler         | Batch formation, priority queuing                  |
| `test_uspp_e2e.cpp`        | USPP end-to-end        | Full scheduling pipeline                           |
| `test_sampling.cpp`        | Sampling               | Temperature, top-P, top-K                          |
| `test_cache_policy.cpp`    | Cache eviction policy  | LRU, frequency-based                               |
| `test_expert_index.cpp`    | Expert index           | Lookup, insertion                                  |
| `test_system_monitor.cpp`  | System monitor         | Metrics collection                                 |
| `test_metal_compute.mm`    | Metal compute          | GPU dispatch, buffer binding                       |
| `test_placeholder.cpp`     | Placeholder            | Stub tests                                         |

**How to add a new unit test:**

1. Create `tests/unit/test_<module>.cpp`.
2. Include the headers you want to test.
3. Define a `CHECK` or `MUGEN_CHECK` macro (or copy from an existing test).
4. Write test functions and call them from `main()`.
5. Re-run `cmake --build build` — CMake auto-discovers new `.cpp` files.

### Integration Tests (`tests/integration/`)

Python-based end-to-end tests that start a real `mugen-server` process and
verify HTTP endpoints. Uses only the Python standard library (`unittest`,
`http.client`, `subprocess`).

See [`tests/integration/README.md`](integration/README.md) for details.

### Fuzz Tests (`tests/fuzz/`)

LLVM libFuzzer targets for security-sensitive parsers. Build with
`-DMUGEN_BUILD_FUZZ=ON` (requires Clang).

| Target              | Module                | What it fuzzes                      |
|---------------------|-----------------------|--------------------------------------|
| `fuzz_gguf_parser`  | GGUF file parser      | Random bytes → `GGUFParser::parse`  |
| `fuzz_tokenizer`    | Tokenizer             | Random UTF-8 → encode/decode        |
| `fuzz_http_parser`  | HTTP request parser   | Random bytes → `parse_request`      |

**Running a fuzz target:**

```bash
# Run for 60 seconds
./build-fuzz/tests/fuzz/fuzz_gguf_parser -max_total_time=60

# With a seed corpus
mkdir corpus && cp tests/fixtures/*.gguf corpus/
./build-fuzz/tests/fuzz/fuzz_gguf_parser corpus/ -max_total_time=300

# Save crashes
./build-fuzz/tests/fuzz/fuzz_http_parser -artifact_prefix=crashes/
```

## Coverage

Target: **80%+ line coverage** for core modules (model parsing, tokenizer,
HTTP server, scheduler).

Generate a coverage report with:

```bash
cmake -B build-cov -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="--coverage" -DCMAKE_EXE_LINKER_FLAGS="--coverage"
cmake --build build-cov -j$(sysctl -n hw.ncpu)
cd build-cov && ctest --output-on-failure
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
open coverage_report/index.html
```

## CI Integration

The GitHub Actions workflow (`.github/workflows/build.yml`) runs unit tests on
every push. Integration and fuzz tests can be added as separate CI jobs.
