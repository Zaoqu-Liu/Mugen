# Metal Kernel Reference

> Complete reference for Mugen's 29 Metal Shading Language compute kernels.

All kernels are defined as compile-time string constants in `src/metal/kernel_sources.h` and compiled at runtime via `MetalCompute::compile_library()`. They target Metal Shading Language 3.0 and are optimized for Apple Silicon SIMD width (32 threads per simdgroup).

## Kernel Index

| # | Kernel | Category | Grid Strategy |
|---|--------|----------|---------------|
| 1 | `matvec_f16` | Matrix-Vector | 1 tg/row, 32 threads |
| 2 | `matvec_q4_0` | Matrix-Vector | 4 rows/tg, 2 simdgroups (64 threads) |
| 3 | `matvec_q8_0` | Matrix-Vector | 1 tg/row, 256 threads (8 simdgroups) |
| 4 | `matvec_q4_k` | Matrix-Vector | 1 tg/row, 256 threads (8 simdgroups) |
| 5 | `matmul_f16` | Matrix-Matrix | 64×32 tile, 128 threads (4 simdgroups) |
| 6 | `matmul_q4_0` | Matrix-Matrix | 64×32 tile, 128 threads (4 simdgroups) |
| 7 | `matmul_f16_naive` | Matrix-Matrix | 1 thread/element |
| 8 | `matmul_q4_0_naive` | Matrix-Matrix | 1 thread/element |
| 9 | `attention_decode` | Attention | 1 tg/head, 256 threads |
| 10 | `prefill_attention` | Attention | 1 tg/(token,head), 256 threads |
| 11 | `flash_attention_decode` | Attention | 1 tg/head, 256 threads (8 simdgroups) |
| 12 | `flash_prefill_attention` | Attention | 1 tg/(token,head), 256 threads |
| 13 | `softmax` | Normalization | 1 tg/row |
| 14 | `rms_norm` | Normalization | 1 tg/row |
| 15 | `silu` | Activation | 1 thread/element |
| 16 | `silu_mul` | Activation | 1 thread/element |
| 17 | `elementwise_add` | Elementwise | 1 thread/element |
| 18 | `elementwise_mul` | Elementwise | 1 thread/element |
| 19 | `rope` | Positional | 1 thread/(head,pair) |
| 20 | `batch_rope` | Positional | 1 thread/(token,head,pair) |
| 21 | `dequantize_q4_0` | Quantization | 1 thread/block |
| 22 | `dequantize_q8_0` | Quantization | 1 thread/block |
| 23 | `dequantize_q4_k` | Quantization | 1 thread/block |
| 24 | `moe_gate` | Routing | single thread |
| 25 | `moe_gate_grouped` | Routing | single thread |
| 26 | `moe_reduce` | Routing | 1 thread/dim |
| 27 | `embedding_lookup` | Utility | 1 thread/element |
| 28 | `scatter_kv` | Utility | 1 thread/element |
| 29 | `argmax_f16` | Utility | 1 tg, 1024 threads |
| — | `bias_broadcast` | Utility | 1 thread/element |

> Note: `bias_broadcast` is included in `kAllKernelsSource` but shares a slot count with the 29 named kernels, as it was added alongside the MLA support.

---

## Matrix-Vector Kernels

### `matvec_f16`

Half-precision matrix-vector multiply. Computes `y = A × x` where A is M×K and x is K×1.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| A (matrix) | `buffer(0)` | `device const half*` | Weight matrix, row-major M×K |
| x (vec) | `buffer(1)` | `device const half*` | Input vector, K elements |
| y (output) | `buffer(2)` | `device half*` | Output vector, M elements |
| M | `buffer(3)` | `constant uint&` | Number of rows |
| K | `buffer(4)` | `constant uint&` | Number of columns |

**Grid:** `{M, 1, 1}`, **Group:** `{32, 1, 1}` (1 simdgroup/row)

Each thread accumulates a strided dot product (stride 32), then `simd_sum()` reduces across the simdgroup. Lane 0 writes the result.

### `matvec_q4_0`

Fused dequantize + matvec from Q4_0 quantized weights. Multi-row: 4 rows per threadgroup, 2 simdgroups.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| matrix_q4 | `buffer(0)` | `device const uint8_t*` | Q4_0 quantized matrix |
| vec | `buffer(1)` | `device const half*` | Input vector (FP16) |
| output | `buffer(2)` | `device half*` | Output vector |
| M | `buffer(3)` | `constant uint&` | Number of rows |
| K | `buffer(4)` | `constant uint&` | Number of columns |

**Grid:** `{ceil(M/4), 1, 1}`, **Group:** `{64, 1, 1}` (2 simdgroups × 32 lanes)

Uses `uint16` word loads and pre-scaled vector trick for efficient nibble extraction. Each simdgroup handles 2 rows; 16 blocks processed per lane per iteration.

### `matvec_q8_0`

Fused dequantize + matvec from Q8_0 quantized weights.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| matrix | `buffer(0)` | `device const uint8_t*` | Q8_0 quantized matrix |
| vec | `buffer(1)` | `device const half*` | Input vector (FP16) |
| output | `buffer(2)` | `device half*` | Output vector |
| M | `buffer(3)` | `constant uint&` | Number of rows |
| K | `buffer(4)` | `constant uint&` | Number of columns |

**Grid:** `{M, 1, 1}`, **Group:** `{256, 1, 1}` (8 simdgroups)

Fully unrolled inner loop over 32 quants per block. Two-level reduction: `simd_sum()` → threadgroup shared memory.

### `matvec_q4_k`

Fused dequantize + matvec from Q4_K quantized weights (Super-block format: 144 bytes per 256 elements).

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| matrix | `buffer(0)` | `device const uint8_t*` | Q4_K quantized matrix |
| vec | `buffer(1)` | `device const half*` | Input vector (FP16) |
| output | `buffer(2)` | `device half*` | Output vector |
| M | `buffer(3)` | `constant uint&` | Number of rows |
| K | `buffer(4)` | `constant uint&` | Number of columns |

**Grid:** `{M, 1, 1}`, **Group:** `{256, 1, 1}` (8 simdgroups)

Decodes 4 sub-blocks of 64 elements each with per-sub-block scale and minimum. Full loop unrolling with `#pragma clang loop unroll(full)`.

---

## Matrix-Matrix Kernels

### `matmul_f16`

Tiled GEMM: `C = B × A^T` (B: N×K, A: M×K row-major, C: N×M column-major output).

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| A | `buffer(0)` | `device const half*` | Weight matrix M×K |
| B | `buffer(1)` | `device const half*` | Input batch N×K |
| C | `buffer(2)` | `device half*` | Output N×M |
| M | `buffer(3)` | `constant uint&` | Weight rows |
| K_dim | `buffer(4)` | `constant uint&` | Inner dimension |
| N | `buffer(5)` | `constant uint&` | Batch size |

**Grid:** `{ceil(M/64), ceil(N/32), 1}`, **Group:** `{128, 1, 1}` (4 simdgroups)

Uses `simdgroup_float8x8` MMA instructions with 8KB shared memory overlay (sa/sb share storage with tc). 64×32 output tile, K-tile=32. Threadgroup memory stride 33 (not 32) to avoid bank conflicts. Vectorized `half4` stores for interior tiles.

### `matmul_q4_0`

Same tiling as `matmul_f16` but with on-the-fly Q4_0 dequantization into threadgroup memory. Cooperative dequantization: each thread decodes 8 nibble pairs.

### `matmul_f16_naive` / `matmul_q4_0_naive`

Fallback kernels with 1 thread per output element. Used for small matrices where tiled GEMM overhead exceeds compute. Uses `float4` vectorized loads with 4-wide accumulation.

---

## Attention Kernels

### `attention_decode`

Single-query decode attention with GQA/MQA support. One threadgroup per Q head.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| Q | `buffer(0)` | `device const half*` | Query, n_heads × head_dim |
| K_cache | `buffer(1)` | `device const half*` | Key cache, seq × kv_stride |
| V_cache | `buffer(2)` | `device const half*` | Value cache, seq × kv_stride |
| output | `buffer(3)` | `device half*` | Output, n_heads × head_dim |
| n_heads | `buffer(4)` | `constant uint&` | Total Q heads |
| n_kv_heads | `buffer(5)` | `constant uint&` | KV heads (GQA) |
| head_dim | `buffer(6)` | `constant uint&` | Dimension per head |
| seq_len | `buffer(7)` | `constant uint&` | Current sequence length |

**Grid:** `{n_heads × 256, 1, 1}`, **Group:** `{256, 1, 1}`

Three-phase: (1) QK^T dot products into threadgroup `scores[8160]`, (2) numerically stable softmax via two-pass simd+tg reduction, (3) V weighted sum with register accumulation → simd → threadgroup reduction.

**Limitation:** Max seq_len = 8160 (threadgroup memory bound). Use `flash_attention_decode` for longer sequences.

### `flash_attention_decode`

Online-softmax flash attention for decode. No sequence length limit.

Same buffer layout as `attention_decode`.

**Grid:** `{n_heads × 256, 1, 1}`, **Group:** `{256, 1, 1}` (8 simdgroups)

Each simdgroup processes a strided subset of KV positions with running max/sum. Register-level Q (4 dims × 32 lanes = 128 dims). After all positions: threadgroup reduction of per-simdgroup `(max, sum, acc[])` using the log-sum-exp rescaling trick.

### `prefill_attention`

Batched prefill attention with causal mask. One threadgroup per (token, head) pair.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| Q_batch | `buffer(0)` | `device const half*` | N × n_heads × head_dim |
| K_cache | `buffer(1)` | `device const half*` | Key cache |
| V_cache | `buffer(2)` | `device const half*` | Value cache |
| O_batch | `buffer(3)` | `device half*` | Output batch |
| n_heads | `buffer(4)` | `constant uint&` | Total Q heads |
| n_kv_heads | `buffer(5)` | `constant uint&` | KV heads |
| head_dim | `buffer(6)` | `constant uint&` | Head dimension |
| kv_len | `buffer(7)` | `constant uint&` | Total KV length |
| start_position | `buffer(8)` | `constant uint&` | Position offset |
| N | `buffer(9)` | `constant uint&` | Number of tokens in batch |

**Grid:** `{N × n_heads × 256, 1, 1}`, **Group:** `{256, 1, 1}`

Causal mask: `causal_len = start_position + token_idx + 1`. Same three-phase approach as `attention_decode`.

### `flash_prefill_attention`

Tiled flash attention for prefill. Processes KV in tiles of 256 with online softmax.

Same buffer layout as `prefill_attention`.

Supports arbitrary sequence lengths. Uses 1024-element tile memory + 32-element reduce buffer. Running `(max, sum, acc[4])` state maintained across tiles with proper rescaling.

---

## Normalization Kernels

### `softmax`

Numerically stable softmax. One threadgroup per row.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| input | `buffer(0)` | `device const half*` | Input rows |
| output | `buffer(1)` | `device half*` | Output probabilities |
| N | `buffer(2)` | `constant uint&` | Row width |

Three-pass: (1) find max via simd+tg reduction, (2) compute exp(x - max) and sum, (3) normalize.

### `rms_norm`

RMSNorm: `y[i] = (x[i] / rms(x)) * weight[i]` where `rms(x) = sqrt(mean(x²) + eps)`.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| input | `buffer(0)` | `device const half*` | Input rows |
| weight | `buffer(1)` | `device const half*` | Learnable scale, 1×N |
| output | `buffer(2)` | `device half*` | Output rows |
| N | `buffer(3)` | `constant uint&` | Row width |
| eps | `buffer(4)` | `constant float&` | Epsilon (typically 1e-5 or 1e-6) |

Two-pass: (1) sum of squares via simd+tg reduction, (2) `rsqrt()` and fused scale.

---

## Activation Kernels

### `silu`

SiLU (Sigmoid Linear Unit): `y = x / (1 + exp(-x))`.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| input | `buffer(0)` | `device const half*` | Input |
| output | `buffer(1)` | `device half*` | Output |
| N | `buffer(2)` | `constant uint&` | Element count |

### `silu_mul`

Fused SiLU + elementwise multiply: `output = SiLU(gate) * up`. Replaces separate `silu` + `elementwise_mul` for FFN gate paths.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| gate | `buffer(0)` | `device const half*` | Gate projection |
| up | `buffer(1)` | `device const half*` | Up projection |
| output | `buffer(2)` | `device half*` | Output |
| N | `buffer(3)` | `constant uint&` | Element count |

---

## Positional Kernels

### `rope`

NeoX-style Rotary Position Embedding (halved pairs). In-place on half-precision vectors.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| x | `buffer(0)` | `device half*` | Input/output (in-place) |
| head_dim | `buffer(1)` | `constant uint&` | Dimension per head |
| n_heads | `buffer(2)` | `constant uint&` | Number of heads |
| position | `buffer(3)` | `constant uint&` | Token position |
| theta_base | `buffer(4)` | `constant float&` | RoPE base (e.g., 10000.0) |

**Grid:** `{n_heads × (head_dim/2), 1, 1}`

Pair rotation: `(x[i], x[i + half_dim])` rotated by `(cos(θ), sin(θ))` where `θ = position / θ_base^(2i/d)`.

### `batch_rope`

Multi-token RoPE for prefill. Extends `rope` with sequential positions.

Additional buffers: `start_pos` (uint), `n_tokens` (uint).

**Grid:** `{n_tokens × n_heads × (head_dim/2), 1, 1}`

---

## Quantization Kernels

### `dequantize_q4_0`

Q4_0 block dequantization: 18 bytes → 32 half-precision values.

Block layout: `[f16 scale (2B)][16 × uint8 nibble pairs (16B)]`

Value reconstruction: `(nibble - 8) × scale`

### `dequantize_q8_0`

Q8_0 block dequantization: 34 bytes → 32 half-precision values.

Block layout: `[f16 scale (2B)][32 × int8 (32B)]`

Value reconstruction: `quant × scale`

### `dequantize_q4_k`

Q4_K super-block dequantization: 144 bytes → 256 half-precision values.

Block layout: `[f16 d (2B)][f16 dmin (2B)][12B packed scales][128B 4-bit quants]`

4 sub-blocks × 64 elements each with per-sub-block scale and minimum.

---

## Routing Kernels

### `moe_gate`

Standard MoE top-K expert selection with softmax normalization.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| gate_logits | `buffer(0)` | `device const half*` | Router logits, n_experts |
| top_indices | `buffer(1)` | `device uint*` | Selected expert indices, top_k |
| top_weights | `buffer(2)` | `device half*` | Normalized weights, top_k |
| n_experts | `buffer(3)` | `constant uint&` | Total expert count |
| top_k | `buffer(4)` | `constant uint&` | Number to select |

**Grid:** `{1, 1, 1}`, **Group:** `{1, 1, 1}` (single thread)

Softmax over all experts, then iterative top-K selection.

### `moe_gate_grouped`

DeepSeek V3 grouped routing: sigmoid → group scoring → group selection → global top-K → renormalization.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| gate_logits | `buffer(0)` | `device const half*` | Router logits |
| top_indices | `buffer(1)` | `device uint*` | Selected indices |
| top_weights | `buffer(2)` | `device half*` | Normalized weights |
| router_bias | `buffer(3)` | `device const half*` | Optional bias |
| n_experts | `buffer(4)` | `constant uint&` | Total experts (e.g., 256) |
| top_k | `buffer(5)` | `constant uint&` | Experts to select (e.g., 8) |
| n_group | `buffer(6)` | `constant uint&` | Number of groups (e.g., 8) |
| topk_group | `buffer(7)` | `constant uint&` | Groups to select (e.g., 4) |
| has_bias | `buffer(8)` | `constant uint&` | Whether router_bias is valid |
| scaling_factor | `buffer(9)` | `constant float&` | Weight scaling (e.g., 2.5) |

Steps: (1) sigmoid activation, (2) per-group top-2 sum scoring, (3) select topk_group groups, (4) global top-K within selected groups (bias shifts selection, not weights), (5) renormalize with scaling_factor.

### `moe_reduce`

Weighted sum of K expert outputs.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| expert_outputs | `buffer(0)` | `device const half*` | K × dim interleaved |
| weights | `buffer(1)` | `device const half*` | K weights |
| output | `buffer(2)` | `device half*` | Reduced output, dim |
| dim | `buffer(3)` | `constant uint&` | Output dimension |
| K | `buffer(4)` | `constant uint&` | Number of experts |

**Grid:** `{dim, 1, 1}`

---

## Utility Kernels

### `embedding_lookup`

Fetches embedding rows by token ID.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| table | `buffer(0)` | `device const half*` | Embedding table, vocab × embed_dim |
| token_ids | `buffer(1)` | `device const uint*` | Token IDs |
| output | `buffer(2)` | `device half*` | n_tokens × embed_dim output |
| embed_dim | `buffer(3)` | `constant uint&` | Embedding dimension |
| n_tokens | `buffer(4)` | `constant uint&` | Number of tokens |

### `scatter_kv`

GPU-side copy from temporary K/V buffer into persistent KV cache.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| src | `buffer(0)` | `device const half*` | Source buffer |
| dst | `buffer(1)` | `device half*` | Destination (KV cache) |
| offset | `buffer(2)` | `constant uint&` | Write offset in dst |
| count | `buffer(3)` | `constant uint&` | Number of elements |

### `argmax_f16`

GPU-side argmax for greedy decoding. Single threadgroup reduces to find the index of the maximum element.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| input | `buffer(0)` | `device const half*` | Logits, N elements |
| result | `buffer(1)` | `device uint*` | Output index (single uint) |
| N | `buffer(2)` | `constant uint&` | Vocab size |

**Grid:** `{1024, 1, 1}`, **Group:** `{1024, 1, 1}`

Two-level reduction with `simd_max` + `simd_shuffle_down` to propagate both value and index.

### `bias_broadcast`

Add 1D bias to each row of a 2D batch buffer (in-place): `batch[gid] += bias[gid % dim]`.

| Buffer | Binding | Type | Description |
|--------|---------|------|-------------|
| batch | `buffer(0)` | `device half*` | Input/output (in-place) |
| bias | `buffer(1)` | `device const half*` | Bias vector, dim elements |
| dim | `buffer(2)` | `constant uint&` | Bias dimension |

---

## Quantization Format Support Matrix

| Format | Block Size | Bytes/Block | Elements/Block | Dequant Kernel | Fused Matvec | Fused Matmul |
|--------|-----------|-------------|----------------|----------------|-------------|-------------|
| F16 | — | 2 | 1 | N/A | `matvec_f16` | `matmul_f16` |
| Q4_0 | 18 B | 18 | 32 | `dequantize_q4_0` | `matvec_q4_0` | `matmul_q4_0` |
| Q8_0 | 34 B | 34 | 32 | `dequantize_q8_0` | `matvec_q8_0` | — |
| Q4_K | 144 B | 144 | 256 | `dequantize_q4_k` | `matvec_q4_k` | — |

The fused matvec kernels (operating directly on quantized data without materialization) are the primary compute path during decode. Matmul kernels are used during prefill for batched token processing.
