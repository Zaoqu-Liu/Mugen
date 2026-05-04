# Mugen Architecture

> Deep dive into the internals of a MoE extreme inference engine for Apple Silicon.

## System Overview

Mugen is a from-scratch C++23 inference engine that runs 300B+ Mixture-of-Experts models on a single Apple Silicon Mac by treating NVMe SSD as a transparent extension of unified memory. The core innovation — **USPP (Unified Speculative-Prefetch Pipeline)** — hides SSD latency behind GPU compute through speculative expert prefetching.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Mugen Inference Engine                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          CLI / HTTP Server                              │ │
│  │  mugen chat | bench | serve | doctor | info | list | pull | rm         │ │
│  └────────┬─────────────────────────┬──────────────────────────────────────┘ │
│           │                         │                                        │
│  ┌────────▼───────────┐    ┌────────▼───────────┐                            │
│  │   USPP Scheduler   │    │   HTTP API Server   │                           │
│  │                     │    │  (OpenAI-compat)    │                           │
│  │  • Draft speculation│    │  /v1/chat/completions│                          │
│  │  • Route prediction │    │  /v1/models          │                          │
│  │  • Adaptive K       │    │  /v1/metrics         │                          │
│  │  • I/O thread mgmt  │    │  /health             │                          │
│  └────────┬────────────┘    └────────┬─────────────┘                         │
│           │                          │                                       │
│  ┌────────▼──────────────────────────▼─────────────────────────────────────┐ │
│  │                     TransformerModel                                    │ │
│  │                                                                         │ │
│  │  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌───────────────────┐   │ │
│  │  │ Embedding │→ │  N layers  │→ │  RMSNorm  │→ │ Logits (LM head) │   │ │
│  │  │  Lookup   │  │  (decode)  │  │  (final)  │  │                   │   │ │
│  │  └───────────┘  └─────┬──────┘  └───────────┘  └───────────────────┘   │ │
│  │                       │                                                 │ │
│  │          ┌────────────┼────────────┐                                    │ │
│  │          ▼            ▼            ▼                                     │ │
│  │    ┌──────────┐ ┌──────────┐ ┌──────────┐                               │ │
│  │    │  Dense   │ │   MoE    │ │   MLA    │                               │ │
│  │    │  FFN     │ │  Router  │ │ Absorbed │                               │ │
│  │    │(SiLU+Mul)│ │+ Experts │ │Attention │                               │ │
│  │    └──────────┘ └──────────┘ └──────────┘                               │ │
│  └─────────────────────────┬───────────────────────────────────────────────┘ │
│                            │                                                 │
│  ┌─────────────────────────▼───────────────────────────────────────────────┐ │
│  │                      Metal Compute                                      │ │
│  │                                                                         │ │
│  │  29 kernels: matvec (F16/Q4_0/Q8_0/Q4_K) · matmul (F16/Q4_0)         │ │
│  │  attention (decode/prefill/flash×2) · softmax · rms_norm · silu        │ │
│  │  rope · moe_gate · moe_reduce · scatter_kv · argmax · embedding · ...  │ │
│  │                                                                         │ │
│  │  Dispatch modes: single | batch | mega-chain (~400 groups/token)       │ │
│  └─────────────────────────┬───────────────────────────────────────────────┘ │
│                            │                                                 │
│  ┌──────────┬──────────────┼──────────────┬──────────────┐                   │
│  │          ▼              ▼              ▼              │                   │
│  │  ┌──────────────┐ ┌──────────┐ ┌──────────────┐      │                   │
│  │  │ Buffer Mgr   │ │ KV Cache │ │  Mmap Loader │      │  Memory Layer     │
│  │  │ (A/B/Pinned) │ │ (4-bit Q)│ │  (zero-copy) │      │                   │
│  │  └──────────────┘ └──────────┘ └──────────────┘      │                   │
│  │          │                             │              │                   │
│  │  ┌───────▼─────────────────────────────▼────────────┐ │                   │
│  │  │              Expert Index + Cache Policy          │ │                   │
│  │  │  heat scoring · co-occurrence · topic shift       │ │                   │
│  │  └──────────────────────────────────────────────────┘ │                   │
│  └───────────────────────────────────────────────────────┘                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                        System Monitor                                    ││
│  │  memory pressure · thermal · degradation levels (Normal→Emergency)       ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
                │                    │                    │
                ▼                    ▼                    ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │   GPU VRAM   │   │ Metal Buffer │   │  NVMe SSD    │
        │  (resident)  │   │  (wired UMA) │   │  (mmap/SSD)  │
        └──────────────┘   └──────────────┘   └──────────────┘
              Tier 1             Tier 2             Tier 3
```

## Core Module Design

### USPP Pipeline

The **Unified Speculative-Prefetch Pipeline** is Mugen's reason for existence. It solves a fundamental problem: a 671B MoE model at Q4 quantization occupies ~340 GB — far exceeding the 64 GB unified memory of a Mac mini M4 Pro.

USPP fuses three techniques into a single pipeline:

1. **Speculative decoding** — A small draft model generates K candidate tokens
2. **Route prediction** — Draft model router logits predict which experts the target model will activate
3. **NVMe prefetch** — Expert weights are read from SSD 2+ layers ahead of need

```
                     USPP Decode Loop (steady state)

  ┌─────────────────────────────────────────────────────────────────┐
  │ Step 1: Draft Model → K candidate tokens                       │
  │         Extract router logits per layer                         │
  └──────────────────────┬──────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────┐
  │ Step 2: Route Predictor → target ExpertKeys (priority-sorted)  │
  │         draft_layer → target_layer mapping                      │
  │         draft_expert → target_expert bucket mapping              │
  │         Deduplicate + sort by descending probability             │
  └──────────────────────┬──────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────┐
  │ Step 3: Async Prefetch → I/O thread reads from mmap/SSD        │
  │         stage_expert() into staging buffer                      │
  │         prefetch_expert() via madvise(MADV_WILLNEED)            │
  │                                                                  │
  │         GPU compute for layer N runs concurrently                │
  └──────────────────────┬──────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────┐
  │ Step 4: swap_buffers() → staging becomes active                 │
  │         Target model verifies K+1 positions                     │
  └──────────────────────┬──────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────┐
  │ Step 5: Speculative Sampling (Leviathan et al. 2023)            │
  │         Accept: p_draft ≤ p_target                               │
  │         Reject: sample from correction distribution              │
  │         Bonus token from target's distribution at last position  │
  └──────────────────────┬──────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────┐
  │ Step 6: Adaptive K Control                                      │
  │         EMA of accept_rate → expand or shrink draft window       │
  │         Mode switching: Normal → Conservative → Emergency        │
  └─────────────────────────────────────────────────────────────────┘
```

**Adaptive control** adjusts the draft window K based on acceptance rate. When cache hit rate drops (topic shift), the scheduler can fall back to Conservative mode (smaller K, fewer prefetches) or Emergency mode (pure autoregressive).

**Key files:**
- `src/core/scheduler/uspp_scheduler.h` — Pipeline orchestration
- `src/core/scheduler/route_predictor.h` — Draft→target expert mapping
- `src/core/scheduler/sampling.h` — Speculative sampling (accept/reject/correction)

### Metal Compute

The `MetalCompute` class is the GPU abstraction layer. It manages the Metal device, compiles shader libraries from embedded source strings, and dispatches compute kernels.

**Dispatch modes:**
- `dispatch_sync()` — Single kernel execution with GPU timestamp profiling
- `dispatch_batch_sync()` — Multiple independent kernels in one command buffer
- `dispatch_chain_sync()` — Grouped dispatches with barriers between groups (mega-chain)

**Mega-chain decode** submits all transformer layers in a single `dispatch_chain_sync()` call (~400 dispatch groups per token). This eliminates per-layer command buffer overhead (~30 μs/CB), yielding a +20% decode speedup on 7B models.

**Buffer management:**
- `create_buffer()` — GPU-only or shared `MTLBuffer`
- `create_buffer_nocopy()` — Zero-copy wrapping of existing memory (mmap regions)
- `create_buffer_from_data()` — Upload from CPU

**Key files:**
- `src/core/compute/metal_compute.h` — Public interface
- `src/core/compute/metal_compute.mm` — Objective-C++ implementation
- `src/metal/kernel_sources.h` — All 29 kernel source strings

### Memory Hierarchy

Mugen uses a three-tier memory hierarchy designed for MoE models that exceed physical RAM:

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier 1: GPU Resident                      │
│  ─────────────────────────────────────────────────────────── │
│  Metal MTLBuffers (wired, not pageable)                      │
│  Contents: embedding table, layer norms, attention weights,  │
│            shared expert weights, KV cache buffers            │
│  Lifetime: allocated at model load, freed at shutdown         │
│  Typical size: 2-8 GB                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Tier 2: Metal Buffer Pool                 │
│  ─────────────────────────────────────────────────────────── │
│  Double-buffered ExpertBuffers (A/B) + Pinned buffer         │
│  Contents: active MoE expert weights for current inference   │
│  Lifecycle: prefetch → stage → swap → consume → evict        │
│  A/B capacity: 15 GB each (configurable)                     │
│  Pinned capacity: 5 GB (hot experts that survive swaps)      │
│  System reserve: 8 GB (for OS + KV cache)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Tier 3: NVMe SSD (via mmap)               │
│  ─────────────────────────────────────────────────────────── │
│  Memory-mapped GGUF model files (MAP_PRIVATE | PROT_READ)    │
│  Contents: all model weights (cold storage)                  │
│  Access: page faults trigger SSD reads (~100 μs/page)        │
│  Management: madvise(MADV_WILLNEED) for prefetch             │
│              madvise(MADV_DONTNEED) for eviction              │
│  Size: up to 340 GB+ (DeepSeek V3 Q4)                       │
└─────────────────────────────────────────────────────────────┘
```

**Double-buffering:** The `BufferManager` maintains two `ExpertBuffer`s (A and B). While the GPU reads from the **active** buffer, the I/O thread fills the **staging** buffer with next-step experts. `swap_buffers()` atomically promotes staging → active.

**Pinned experts:** Frequently-accessed experts (high heat score) are promoted to a separate pinned buffer that survives swaps. The `ExpertIndex` tracks heat scores using a weighted combination of access frequency, recency, and a sliding-window bonus.

**Key files:**
- `src/core/memory/buffer_manager.h` — Double-buffer + pinned management
- `src/core/memory/mmap_loader.h` — RAII mmap with madvise
- `src/core/memory/kv_cache.h` — Quantized KV cache

### MoE Routing

Mugen supports four MoE routing paths:

| Path | Architecture | Routing Method |
|------|-------------|----------------|
| Standard softmax | OLMoE, Mixtral | Top-K from softmax(gate_logits) |
| Grouped sigmoid | DeepSeek V3 | sigmoid → group top-2 → top-N groups → global top-K → renorm |
| Shared + routed | DeepSeek V2/V3 | Shared expert (always active) + routed experts |
| Dense fallback | First-K layers | Dense FFN for early layers (DeepSeek V3's `first_k_dense_replace`) |

The `moe_gate` kernel handles standard softmax routing. The `moe_gate_grouped` kernel implements DeepSeek V3's grouped routing with optional router bias. Both run on GPU as single-thread kernels (routing 256 experts is trivial compute).

After routing, `moe_reduce` computes the weighted sum of expert outputs.

### Speculative Decoding

Implementation follows Leviathan et al. 2023 ("Fast Inference from Transformers via Speculative Decoding"):

1. Draft model generates K tokens with their probability distributions
2. Target model verifies all K positions in a single forward pass (parallel)
3. For each position i: accept if `random_uniform < min(1, p_target[i] / p_draft[i])`
4. On first rejection at position j: sample from `max(0, p_target[j] - p_draft[j])` (normalized)
5. Bonus: sample one extra token from target's distribution at position K (or last accepted + 1)

This guarantees the output distribution is identical to pure target-model sampling.

**Key parameters:**
- `draft_k`: 2-8 (adaptive based on acceptance rate)
- `accept_threshold`: 0.8 (target for EMA accept rate)
- `warmup_tokens`: 50 (pure autoregressive before switching to speculative)

### MLA Absorbed Attention

Multi-head Latent Attention (DeepSeek V2/V3) compresses the KV cache by projecting K and V through a low-rank bottleneck:

```
Standard Attention:             MLA Absorbed Attention:
  KV per token = 2 × n_kv_heads × head_dim    KV per token = kv_lora_rank + qk_rope_head_dim
  DeepSeek V3: 2 × 128 × 192 = 49152 dims     DeepSeek V3: 512 + 64 = 576 dims
                                                Compression ratio: 42×
```

The absorbed attention fuses the KV up-projection into the attention computation, avoiding materialization of the full KV. During decode, the compressed KV representation is stored in the cache; during prefill, a batched CPU attention path handles the larger matrices.

## Data Flow: Token Lifecycle

A single token's journey from input to output:

```
Input token_id: 42
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Embedding Lookup (GPU kernel: embedding_lookup)          │
  │    table[42 * embed_dim .. (42+1) * embed_dim] → hidden     │
  │    Output: half[embed_dim]                                   │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────── For each layer l = 0..N-1 ──────────────────┐
  │                                                              │
  │  ┌───────────────────────────────────────────────────────┐   │
  │  │ 2a. RMSNorm (GPU: rms_norm)                           │   │
  │  │     hidden → attn_input                                │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  │  ┌────────────────────▼──────────────────────────────────┐   │
  │  │ 2b. Attention                                          │   │
  │  │                                                         │   │
  │  │  Q = attn_input × W_q    (matvec_f16 or matvec_q4_0)  │   │
  │  │  K = attn_input × W_k                                  │   │
  │  │  V = attn_input × W_v                                  │   │
  │  │  RoPE(Q, K, position)    (rope or batch_rope)          │   │
  │  │  scatter_kv(K, V → KV_cache[l])                        │   │
  │  │  attn_out = flash_attention_decode(Q, K_cache, V_cache)│   │
  │  │  projected = attn_out × W_o                             │   │
  │  │                                                         │   │
  │  │  [MLA path: compressed KV, absorbed up-projection]      │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  │  ┌────────────────────▼──────────────────────────────────┐   │
  │  │ 2c. Residual Add (GPU: elementwise_add)                │   │
  │  │     hidden = hidden + projected                         │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  │  ┌────────────────────▼──────────────────────────────────┐   │
  │  │ 2d. RMSNorm (GPU: rms_norm)                           │   │
  │  │     hidden → ffn_input                                 │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  │  ┌────────────────────▼──────────────────────────────────┐   │
  │  │ 2e. FFN / MoE                                          │   │
  │  │                                                         │   │
  │  │  [Dense path]                                           │   │
  │  │    gate = ffn_input × W_gate   →  SiLU                 │   │
  │  │    up   = ffn_input × W_up                              │   │
  │  │    ffn_out = (gate ⊙ up) × W_down                      │   │
  │  │                                                         │   │
  │  │  [MoE path]                                             │   │
  │  │    logits = ffn_input × W_router                        │   │
  │  │    (indices, weights) = moe_gate(logits)                │   │
  │  │    for each expert in top-K:                            │   │
  │  │      expert_out = expert_ffn(ffn_input)                 │   │
  │  │    ffn_out = moe_reduce(expert_outputs, weights)        │   │
  │  │                                                         │   │
  │  │  [MoE + Shared Expert path (DeepSeek V2/V3)]           │   │
  │  │    shared_out = shared_expert_ffn(ffn_input)            │   │
  │  │    ffn_out = shared_out + moe_reduce(...)               │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  │  ┌────────────────────▼──────────────────────────────────┐   │
  │  │ 2f. Residual Add (GPU: elementwise_add)                │   │
  │  │     hidden = hidden + ffn_out                           │   │
  │  └────────────────────┬──────────────────────────────────┘   │
  │                       │                                      │
  └───────────────────────┼──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Final RMSNorm (GPU: rms_norm)                            │
  │    hidden → normalized                                       │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. LM Head (GPU: matvec_q4_0 or matvec_f16)                │
  │    logits = normalized × W_lm_head   →  float[vocab_size]   │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. Sampling (CPU)                                           │
  │    temperature_scale → top_p_filter → softmax → sample       │
  │    Output: next_token_id                                     │
  └─────────────────────────────────────────────────────────────┘
```

## Metal Kernel Architecture

All 29 kernels are embedded as compile-time string constants in `src/metal/kernel_sources.h`. They are compiled into a single Metal library at runtime via `MetalCompute::compile_library()`.

### Kernel Categories

| Category | Kernels | Count |
|----------|---------|-------|
| **Matrix-Vector** | `matvec_f16`, `matvec_q4_0`, `matvec_q8_0`, `matvec_q4_k` | 4 |
| **Matrix-Matrix** | `matmul_f16`, `matmul_q4_0`, `matmul_f16_naive`, `matmul_q4_0_naive` | 4 |
| **Attention** | `attention_decode`, `prefill_attention`, `flash_attention_decode`, `flash_prefill_attention` | 4 |
| **Normalization** | `softmax`, `rms_norm` | 2 |
| **Activation** | `silu`, `silu_mul` | 2 |
| **Elementwise** | `elementwise_add`, `elementwise_mul` | 2 |
| **Positional** | `rope`, `batch_rope` | 2 |
| **Quantization** | `dequantize_q4_0`, `dequantize_q8_0`, `dequantize_q4_k` | 3 |
| **Routing** | `moe_gate`, `moe_gate_grouped`, `moe_reduce` | 3 |
| **Utility** | `embedding_lookup`, `scatter_kv`, `argmax_f16`, `bias_broadcast` | 4 |
| | | **29 total** |

### SIMD Optimization Strategy

All performance-critical kernels exploit Apple GPU SIMD width (32 threads per simdgroup):

- **Matvec kernels:** One threadgroup per output row. Threads collaboratively dot-product along K, then `simd_sum()` reduces within the simdgroup. Multi-row variants (4 rows/tg, 2 simdgroups) amortize memory latency.
- **Matmul kernels (tiled GEMM):** 64×32 output tile, 128 threads (4 simdgroups). Uses `simdgroup_float8x8` / `simdgroup_half8x8` matrix multiply-accumulate (MMA) instructions. Threadgroup memory with +1 padding (stride 33) avoids bank conflicts.
- **Attention kernels:** Flash variants use online softmax with tiled KV processing. Each simdgroup processes a subset of sequence positions. Register-level accumulation with `simd_sum()` for Q·K dot products (32 lanes × 4 dims = 128 dims per lane).
- **Reduction kernels (softmax, rms_norm):** Two-level reduction: `simd_sum()`/`simd_max()` within simdgroup, then threadgroup shared memory across simdgroups.

## GGUF Parsing

`GGUFParser` performs zero-copy parsing of GGUF model files:

```
GGUF File Layout:
┌──────────────────────┐
│  Magic: "GGUF"       │  4 bytes
│  Version: 3          │  4 bytes
│  n_tensors           │  8 bytes
│  n_kv                │  8 bytes
├──────────────────────┤
│  KV Metadata         │  Variable-length key-value pairs
│  (architecture,      │  Types: string, int, float, array, ...
│   layer counts,      │
│   expert config,     │
│   tokenizer, ...)    │
├──────────────────────┤
│  Tensor Descriptors  │  Per-tensor: name, dims, type, offset
│  (name, shape,       │
│   dtype, offset)     │
├──────────────────────┤  ← data_offset (aligned to 32 bytes)
│                      │
│  Tensor Data Blob    │  Contiguous tensor weights
│  (mmap target)       │  Accessed via MmapLoader
│                      │
└──────────────────────┘
```

Key capabilities:
- **Sharded GGUF:** Detects split files (`*-00001-of-00005.gguf`) and merges tensor info across shards
- **Expert tensor mapping:** `expert_tensor_names(layer, expert_id)` follows llama.cpp naming: `blk.<L>.ffn_{gate,up,down}_exps.<E>.weight`
- **Architecture detection:** Reads `general.architecture` to configure Llama, Qwen, OLMoE, DeepSeek variants

## KV Cache & Prefix Reuse

### Cache Structure

`KVCache` stores key and value tensors for all layers with optional 4-bit quantization:

- **Quantized mode (default):** Recent tokens (last `fp16_preserve_last = 256`) stored as FP16; older tokens quantized to 4-bit. Achieves ~4× compression vs full FP16.
- **FP16 mode:** All positions stored as half-precision.

Layout: `[n_layers][max_seq_len][n_kv_heads * head_dim]` with token-major ordering for cache-friendly attention access.

### Prefix Reuse

Multi-turn chat reuses KV cache from previous turns:

```
Turn 1: [system] [user_msg_1] [assistant_msg_1]
         ─────── cached KV ────────────────────

Turn 2: [system] [user_msg_1] [assistant_msg_1] [user_msg_2]
         ─────── reused (skip prefill) ────────  ──── new ──
```

The CLI and server both track `prev_prompt_tokens` and compute the longest common prefix. `KVCache::truncate()` trims to the reuse point; only new tokens run through prefill. This dramatically reduces TTFT for multi-turn conversations.

### GPU Scatter

The `scatter_kv` kernel copies freshly computed K/V vectors from temporary GPU buffers into the persistent KV cache, replacing what was previously a CPU memcpy bottleneck.

## Supported Model Architectures

| Architecture | Examples | Attention | FFN | Expert Routing |
|-------------|----------|-----------|-----|---------------|
| Llama | Llama 3.x, Llama 2 | GQA | Dense SiLU-gated | N/A |
| Qwen | Qwen 2.5 | GQA | Dense SiLU-gated | N/A |
| OLMoE | OLMoE-1B-7B | MHA | MoE (64 experts, top-8) | Softmax top-K |
| DeepSeek V2 | DeepSeek-V2-Lite | MLA | MoE + shared expert | Softmax top-K |
| DeepSeek V3 | DeepSeek-V3-671B | MLA | MoE + shared expert | Grouped sigmoid |

## Build System

CMake 3.21+ with C++23 and OBJCXX (Objective-C++ for Metal API). The Metal shader compilation pipeline:

```
*.metal source → xcrun metal -std=metal3.0 -O2 → *.air
*.air files    → xcrun metallib → mugen.metallib
```

Runtime: kernel sources in `kernel_sources.h` are compiled via `MTLDevice::newLibraryWithSource:` into `MTLLibrary` objects. The `.metallib` build is for validation; runtime compilation allows the same binary to work without Xcode installed.

## Thread Model

```
Main Thread          I/O Thread              GPU Command Queue
  │                    │                        │
  │─── generate() ───→ │                        │
  │                    │                        │
  │  run_draft()       │                        │
  │───── GPU dispatch ──────────────────────────→│ draft forward
  │                    │                        │
  │  schedule_prefetch()                        │
  │─── submit_io() ──→│                        │
  │                    │── madvise(WILLNEED)    │
  │                    │── memcpy to staging    │
  │                    │                        │
  │  verify_with_target()                       │
  │───── GPU dispatch ──────────────────────────→│ target forward
  │                    │                        │
  │  swap_buffers()    │                        │
  │  speculative_sample()                       │
  │  adapt_k()         │                        │
  │                    │                        │
  │  [loop]            │                        │
```

The I/O thread runs a task queue (`io_queue_`) with generation-based ordering to allow bulk completion waits. `stop_requested_` (atomic bool) enables graceful shutdown.
