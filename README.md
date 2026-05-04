# Mugen (無限)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://github.com/Zaoqu-Liu/Mugen/actions/workflows/build.yml/badge.svg)](https://github.com/Zaoqu-Liu/Mugen/actions)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://www.apple.com/mac/)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

**有限内存，無限推理 — Finite memory, infinite inference.**

Mugen is a MoE (Mixture-of-Experts) extreme inference engine built for Apple Silicon. It runs 300B+ parameter MoE models on a single Mac mini M4 Pro with 64 GB unified memory by treating NVMe SSD as a transparent extension of VRAM through speculative expert prefetching.

## Why Mugen?

> **"Run DeepSeek V3 (671B) on a consumer Mac mini."**

Existing LLM inference engines (llama.cpp, MLX, Ollama) require the model to fit entirely in unified memory. A 671B parameter MoE model at Q4 quantization occupies ~340 GB — impossible on any consumer Mac.

Mugen's **USPP (Unified Speculative-Prefetch Pipeline)** solves this through three integrated techniques:

1. **Speculative decoding** — a small draft model generates candidate tokens
2. **Route prediction** — draft model router logits predict which experts the target model will activate
3. **NVMe prefetch** — expert weights are read from SSD two layers ahead of need

The result: SSD round-trip latency (~100 μs) is hidden behind GPU compute, making NVMe feel like VRAM.

## Architecture

```
                        USPP Pipeline (steady state)
  ──────────────────────────────────────────────────────────────────

  Layer N          Layer N+1        Layer N+2        Layer N+3
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ GPU      │     │ GPU      │     │ GPU      │     │ GPU      │
  │ Compute  │     │ Compute  │     │ Compute  │     │ Compute  │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘
       │                │                │                │
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Prefetch │     │ Prefetch │     │ Prefetch │
  │ N+1 exp  │     │ N+2 exp  │     │ N+3 exp  │
  └──────────┘     └──────────┘     └──────────┘
       │                │                │
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ SSD Read │     │ SSD Read │     │ SSD Read │
  │ N+2 exp  │     │ N+3 exp  │     │ N+4 exp  │
  └──────────┘     └──────────┘     └──────────┘

  ◄── time ──────────────────────────────────────────────────────►
```

The pipeline maintains three concurrent stages per layer — by the time the GPU needs an expert's weights, they are already resident in a Metal buffer.

## Key Design Points

- **Zero-copy weight access.** Model files are memory-mapped; expert tensors are promoted into wired MTLBuffers on demand and evicted under LRU.
- **Speculative prefetch.** Router logits from layer N predict which experts layer N+2 will need.
- **Quantized Metal kernels.** Fused dequant + GEMV kernels for Q4_0, Q4_K, and Q8_0 formats, tuned for Apple GPU SIMD width.
- **GGUF native.** Reads standard GGUF model files directly — no conversion step required.
- **Mega-chain decode.** All transformer layers in a single Metal dispatch — ~400 groups submitted once per token.
- **MLA absorbed attention.** Compressed KV cache for DeepSeek V2/V3 (576 dims vs 24576 expanded — 42× compression).

## Performance

| Model | TTFT | Prefill | Decode | Notes |
|-------|------|---------|--------|-------|
| Llama-3.2-1B Q4_0 | 69 ms | 694 tok/s | 266.2 tok/s | mega-chain |
| Qwen2.5-7B Q4_0 | 163 ms | 295 tok/s | 56.1 tok/s | mega-chain |
| DeepSeek-V2-Lite Q8_0 | 9.1 s | 5.4 tok/s | 4.9 tok/s | CPU MLA (GPU kernel WIP) |

*Benchmarked on Mac mini M4 Pro (14 CPU / 20 GPU / 64 GB). DeepSeek V2-Lite decode will reach 30+ tok/s once the GPU MLA attention kernel lands.*

## Quick Start

### Requirements

- macOS 14+ (Sonoma or later)
- Apple Silicon Mac with 32 GB+ unified memory (64 GB recommended)
- Xcode (required for Metal shader compilation)
- CMake 3.21+

### System Check

```sh
python3 tools/check_system.py
```

### Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Run

```sh
# Chat with a model
./build/src/mugen-cli chat <model.gguf>

# Benchmark
./build/src/mugen-cli bench <model.gguf>

# Start API server (OpenAI-compatible)
./build/src/mugen-cli serve <model.gguf> --port 8080

# Check system
./build/src/mugen-cli doctor
```

### Run tests

```sh
ctest --test-dir build --output-on-failure
```

## Project Status

### Completed
- [x] Dense inference (0.5B, 3B, 7B)
- [x] MoE inference (OLMoE 64 experts, DeepSeek V2-Lite)
- [x] Flash attention (prefill + decode, SIMD-optimized)
- [x] Mega-chain decode (single dispatch per token)
- [x] KV cache with prefix reuse
- [x] Speculative decoding with draft model
- [x] USPP pipeline architecture
- [x] 29 Metal kernels covering all compute paths
- [x] 16 unit tests, all passing

### In Progress
- [ ] GPU MLA attention kernel (DeepSeek V2/V3 decode performance)
- [ ] Three-tier memory scheduler (resident / buffer / SSD)
- [ ] DeepSeek V3 end-to-end (671B Q4)
- [ ] USPP speculative-prefetch fusion for SSD offload
- [ ] 8+ tok/s decode on DeepSeek V3 (North Star)

See [docs/VISION.md](docs/VISION.md) for the full technical roadmap and North Star.

## Project Structure

```
src/
  core/model/         Transformer model, MLA attention, MoE routing
  core/compute/       Metal GPU dispatch and kernel management
  core/scheduler/     USPP pipeline, sampling, route prediction
  core/memory/        KV cache, buffer management, mmap loader
  core/prefetch/      Expert index, cache policy, prefetch engine
  core/monitor/       System monitoring (memory, thermal)
  metal/              Metal Shading Language kernels (29 kernels)
  model/              GGUF parser, tokenizer (BPE + SentencePiece)
  server/             HTTP API server (OpenAI-compatible)
  cli/                Command-line interface (chat, bench, serve)
include/mugen/        Public headers
tests/unit/           Unit tests (16 suites)
bench/micro/          Micro-benchmarks
tools/                Utility scripts
docs/                 Documentation
```

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, and pull request guidelines. All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Mugen is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Mugen Contributors.
