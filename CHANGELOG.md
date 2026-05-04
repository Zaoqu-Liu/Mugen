# Changelog

All notable changes to Mugen will be documented in this file.

## [0.1.0] - 2026-05-04

### Added
- Initial public release
- USPP (Unified Speculative-Prefetch Pipeline) architecture for MoE inference
- 29 Metal Shading Language kernels (Q4_0, Q4_K, Q8_0, F16 matvec/matmul, softmax, rms_norm, silu, rope, flash attention, MoE routing, KV cache)
- Mega-chain decode: all layers in a single Metal dispatch
- MLA absorbed attention for DeepSeek V2/V3 architecture
- GGUF native model loading (zero conversion required)
- MoE batch routing with scatter_kv GPU kernel
- Speculative decoding with draft model support
- Flash attention (prefill + decode paths)
- KV cache with prefix reuse for multi-turn conversations
- Dense inference path (0.5B, 3B, 7B models verified)
- MoE inference path (OLMoE, DeepSeek-V2-Lite verified)
- OpenAI-compatible HTTP API server scaffold
- CLI: chat, bench, info, doctor, serve, pull, rm, list
- Per-kernel GPU profiling (Metal hardware timestamps)
- 16 unit tests covering all major components
- Zero third-party dependencies (Metal + Foundation + POSIX only)
- C++23 codebase with `-Werror` enforcement
