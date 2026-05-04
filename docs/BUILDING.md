# Building Mugen

> Detailed build guide for the Mugen inference engine.

## Requirements

### Hardware

- **Apple Silicon Mac** (M1, M2, M3, M4 family — any variant)
- **32 GB+ unified memory** recommended (16 GB minimum for small models)
- **64 GB** required for DeepSeek V2-Lite and larger MoE models
- NVMe SSD with sufficient free space for model files

### Software

| Dependency | Minimum Version | Check Command |
|-----------|----------------|---------------|
| macOS | 14.0 (Sonoma) | `sw_vers` |
| Xcode | 15.0 | `xcodebuild -version` |
| CMake | 3.21 | `cmake --version` |
| Apple Clang | C++23 support | `clang++ --version` |
| Python 3 | 3.8+ (for tools) | `python3 --version` |

### System Check

Run the built-in diagnostic tool to verify your environment:

```sh
python3 tools/check_system.py
```

This checks: macOS version, Apple Silicon detection, Xcode installation, Metal compiler availability, available memory, and SSD space.

### Installing Prerequisites

**Xcode** (required for Metal shader compilation):
```sh
# Install from Mac App Store, or:
xcode-select --install

# Verify Metal compiler is available:
xcrun --find metal
```

> **Important:** The full Xcode.app is required — Command Line Tools alone do not include the Metal compiler. If `xcrun --find metal` fails, install Xcode from the App Store.

**CMake:**
```sh
brew install cmake
```

---

## Build Configurations

### Release (recommended for benchmarks and production)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

Enables `-Werror` (warnings as errors) and full optimizations.

### Debug

```sh
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(sysctl -n hw.ncpu)
```

Includes debug symbols, no optimizations. Useful for stepping through code in a debugger.

### Address Sanitizer (ASan)

```sh
cmake -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMUGEN_ENABLE_ASAN=ON
cmake --build build-asan -j$(sysctl -n hw.ncpu)
```

Detects memory errors: buffer overflows, use-after-free, double-free, memory leaks. Expect ~2× slowdown.

### Thread Sanitizer (TSan)

```sh
cmake -B build-tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMUGEN_ENABLE_TSAN=ON
cmake --build build-tsan -j$(sysctl -n hw.ncpu)
```

Detects data races and threading issues. Expect ~5-15× slowdown. Particularly useful for verifying the I/O thread, buffer swap atomics, and the USPP scheduler.

> **Note:** ASan and TSan cannot be enabled simultaneously. The build system will error if both are set to ON.

### Disabling Optional Targets

```sh
# Skip tests
cmake -B build -DMUGEN_BUILD_TESTS=OFF

# Skip benchmarks
cmake -B build -DMUGEN_BUILD_BENCH=OFF
```

---

## Metal Shader Compilation

Metal shaders are compiled through a two-stage pipeline:

### Stage 1: Source → AIR (Apple Intermediate Representation)

```
src/metal/*.metal  →  xcrun metal -std=metal3.0 -O2 -c  →  *.air
```

Each `.metal` file is compiled independently to an `.air` intermediate file.

### Stage 2: AIR → Metallib

```
*.air files  →  xcrun metallib  →  mugen.metallib
```

All `.air` files are linked into a single `mugen.metallib`.

### Runtime Compilation

In practice, Mugen compiles kernels at runtime from source strings embedded in `src/metal/kernel_sources.h` via `MTLDevice::newLibraryWithSource:`. This approach:

- Works without Xcode installed on the target machine
- Allows the same binary to run on different GPU generations
- The `.metallib` build serves as a compile-time validation step

If the Metal compiler is not found (no Xcode), the CMake build warns but continues. Runtime shader compilation will still work as long as the Metal framework is available.

---

## Running Tests

```sh
# Build with tests (default)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(sysctl -n hw.ncpu)

# Run all 16 test suites
ctest --test-dir build --output-on-failure

# Run a specific test
ctest --test-dir build -R test_kv_cache --output-on-failure

# Verbose output
ctest --test-dir build -V
```

### Test Suites

| Test | What it covers |
|------|---------------|
| `test_gguf_parser` | GGUF header, metadata, tensor descriptor parsing |
| `test_tokenizer` | BPE and SentencePiece encode/decode |
| `test_metal_compute` | Metal device init, buffer ops, kernel dispatch |
| `test_kv_cache` | Append, read, truncate, quantization, compression |
| `test_buffer_manager` | Double-buffer swap, pinned experts, memory pressure |
| `test_mmap_loader` | File mapping, page alignment, madvise |
| `test_cache_policy` | Eviction scoring, co-occurrence, topic shift |
| `test_expert_index` | Expert location build, heat scores, pinning |
| `test_route_predictor` | Draft→target layer/expert mapping |
| `test_uspp_scheduler` | Pipeline orchestration, adaptive K |
| `test_uspp_e2e` | End-to-end USPP decode loop |
| `test_sampling` | Temperature, top-p, speculative accept/reject |
| `test_transformer` | Model load from GGUF, forward pass |
| `test_http_server` | HTTP parsing, routing, response formatting |
| `test_system_monitor` | Memory pressure, degradation levels |
| `test_placeholder` | Build system validation |

---

## Running Benchmarks

```sh
# Build benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMUGEN_BUILD_BENCH=ON
cmake --build build -j$(sysctl -n hw.ncpu)

# Micro-benchmarks (in bench/micro/)
./build/bench/micro/bench_ssd_read
./build/bench/micro/bench_metal_buffer
./build/bench/micro/bench_uma_bandwidth
./build/bench/micro/bench_expert_fault
./build/bench/micro/bench_madvise
```

---

## IDE Integration

### Xcode

```sh
cmake -B build-xcode -G Xcode
open build-xcode/mugen.xcodeproj
```

Select the `mugen-cli` scheme to build and run. Metal shaders will be compiled as custom build rules.

### CLion

1. Open the project root directory in CLion
2. CLion auto-detects the `CMakeLists.txt`
3. Set CMake options in **Preferences → Build → CMake**:
   - **CMake options:** `-DCMAKE_BUILD_TYPE=Debug`
   - **Build directory:** `build-clion`
4. Select `mugen-cli` as the run target
5. Add program arguments: `chat <model.gguf>` in the Run Configuration

### VS Code

1. Install the **CMake Tools** extension
2. Open the project root
3. Press `Cmd+Shift+P` → "CMake: Configure"
4. Select the **Clang** kit (Apple Clang)
5. Set build type in the status bar (Debug/Release)
6. Press `Cmd+Shift+P` → "CMake: Build"

For IntelliSense, ensure `compile_commands.json` is generated (enabled by default via `CMAKE_EXPORT_COMPILE_COMMANDS`):

```json
// .vscode/settings.json
{
    "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json"
}
```

---

## FAQ

### Q: Build fails with "Metal compiler not found"

**A:** Install the full Xcode.app from the Mac App Store (not just Command Line Tools). Then run:

```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun --find metal  # Should print a path
```

The build will still succeed without the Metal compiler — shader compilation just won't produce a `.metallib`. Runtime compilation will still work.

### Q: "C++23 not supported" or "expected" header not found

**A:** Ensure Apple Clang 15+ is installed (ships with Xcode 15+). Check:

```sh
clang++ -std=c++23 -x c++ -E - <<< '#include <expected>' 2>&1 | head -1
```

If this fails, update Xcode to version 15 or later.

### Q: Build succeeds but "Metal device not found" at runtime

**A:** This typically happens when running on x86_64 (Intel Mac) or inside a VM. Mugen requires Apple Silicon with Metal 3.0 support. Verify:

```sh
system_profiler SPDisplaysDataType | grep Metal
# Should show "Metal Family: Apple X" (X ≥ 7)
```

### Q: How do I cross-compile or build for a different target?

**A:** Mugen is Apple Silicon-only by design. The CMake will `FATAL_ERROR` on non-Apple platforms. Cross-compilation is not supported.

### Q: Tests pass locally but fail in CI

**A:** Ensure the CI runner:
1. Is macOS with Apple Silicon (arm64)
2. Has Xcode installed (not just Command Line Tools)
3. Has the Metal framework available (not all CI providers support GPU)

The GitHub Actions workflow (`.github/workflows/build.yml`) is configured for `macos-14` runners with Apple Silicon.

### Q: How do I profile kernel performance?

**A:** Use the built-in profiling:

```sh
./build/src/mugen-cli bench <model.gguf>
```

This outputs per-kernel GPU timestamps. For deeper analysis, use Xcode's Metal System Trace (Instruments → Metal System Trace template).

### Q: Build is very slow

**A:** Ensure parallel compilation:

```sh
cmake --build build -j$(sysctl -n hw.ncpu)
```

The `kernel_sources.h` header is large (~3000 lines) and included in several compilation units. Precompiled headers are not currently used but could be added if build times become problematic.
