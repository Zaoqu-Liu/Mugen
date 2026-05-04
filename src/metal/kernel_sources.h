#pragma once

namespace mugen::metal {

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 dequantization
//
// Q4_0 block layout (18 bytes per 32 elements):
//   [f16 scale] [16 × uint8 nibble pairs]
//   Each byte holds two 4-bit unsigned ints. Value = (nibble - 8) * scale.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kDequantQ4_0Source = R"(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint QK4_0 = 32;
constant constexpr uint BLOCK_SIZE_Q4_0 = 18;  // 2 (f16 scale) + 16 (data)

kernel void dequantize_q4_0(
    device const uint8_t* input  [[buffer(0)]],
    device half*          output [[buffer(1)]],
    constant uint&        n_blocks [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_blocks) return;

    device const uint8_t* block = input + tid * BLOCK_SIZE_Q4_0;

    // First 2 bytes: f16 scale stored as raw bits
    half scale = *reinterpret_cast<device const half*>(block);
    device const uint8_t* data = block + 2;

    device half* out = output + tid * QK4_0;

    for (uint j = 0; j < 16; ++j) {
        uint8_t byte = data[j];
        int lo = (int)(byte & 0x0F) - 8;
        int hi = (int)(byte >> 4)    - 8;

        out[j]      = scale * (half)lo;
        out[j + 16] = scale * (half)hi;
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Half-precision matrix-vector multiply (M×K * K×1 = M×1)
//
// Each row assigned to one simdgroup (32 lanes). Each lane accumulates
// a partial dot product over K/32 elements, then simd_sum reduces.
// Multiple simdgroups per threadgroup handle multiple rows.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatvecF16Source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f16(
    device const half* matrix [[buffer(0)]],
    device const half* vec    [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     M      [[buffer(3)]],
    constant uint&     K      [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // One threadgroup per row. Within the threadgroup, all threads
    // collaborate on the dot product via simd_sum reduction.
    uint row = tgid;
    if (row >= M) return;

    device const half* row_ptr = matrix + (uint64_t)row * K;

    float acc = 0.0f;
    for (uint col = tid; col < K; col += 32) {
        acc += float(row_ptr[col]) * float(vec[col]);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[row] = half(acc);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Fused Q4_0 matrix-vector multiply
//
// Directly computes matvec from Q4_0-quantized matrix without materializing
// the dequantized matrix. Each threadgroup handles one row (M dimension).
// Within a threadgroup, threads collaboratively iterate over Q4_0 blocks
// along K, then perform simdgroup + threadgroup reduction.
//
// Matrix layout: M rows, each row has (K/32) Q4_0 blocks stored contiguously.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatvecQ4_0Source = R"(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint QK4_0 = 32;
constant constexpr uint BLOCK_SIZE_Q4_0 = 18;

// Multi-row matvec: 4 rows per threadgroup, 2 simdgroups (64 threads).
// Optimized for Apple Silicon: uint16 loads, half2 vector loads, simd_sum reduction.
kernel void matvec_q4_0(
    device const uint8_t* matrix_q4 [[buffer(0)]],
    device const half*    vec       [[buffer(1)]],
    device half*          output    [[buffer(2)]],
    constant uint&        M         [[buffer(3)]],
    constant uint&        K         [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint N_R0 = 4;
    const uint N_SG = 2;
    const uint rows_per_sg = N_R0 / N_SG;

    uint n_blocks = K / QK4_0;

    for (uint ri = 0; ri < rows_per_sg; ri++) {
        uint row = tgid * N_R0 + sgitg * rows_per_sg + ri;
        if (row >= M) return;

        device const uint8_t* row_data = matrix_q4
            + (uint64_t)row * n_blocks * BLOCK_SIZE_Q4_0;

        float acc = 0.0f;

        for (uint b = tiisg; b < n_blocks; b += 32) {
            device const uint8_t* blk = row_data + b * BLOCK_SIZE_Q4_0;
            half scale = *reinterpret_cast<device const half*>(blk);
            device const uint16_t* qs = reinterpret_cast<device const uint16_t*>(blk + 2);
            device const half* vb = vec + b * QK4_0;

            float block_sum = 0.0f;

            for (uint j = 0; j < 8; j++) {
                uint16_t word = qs[j];
                uint8_t lo_byte = word & 0xFF;
                uint8_t hi_byte = (word >> 8);

                int lo0 = (int)(lo_byte & 0x0F) - 8;
                int hi0 = (int)(lo_byte >> 4) - 8;
                int lo1 = (int)(hi_byte & 0x0F) - 8;
                int hi1 = (int)(hi_byte >> 4) - 8;

                half2 v_lo = *reinterpret_cast<device const half2*>(vb + j * 2);
                half2 v_hi = *reinterpret_cast<device const half2*>(vb + 16 + j * 2);

                block_sum += float(lo0) * float(v_lo[0]);
                block_sum += float(lo1) * float(v_lo[1]);
                block_sum += float(hi0) * float(v_hi[0]);
                block_sum += float(hi1) * float(v_hi[1]);
            }

            acc += float(scale) * block_sum;
        }

        acc = simd_sum(acc);
        if (tiisg == 0) output[row] = half(acc);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Softmax (numerically stable)
//
// Two-pass with simdgroup + threadgroup reduction:
//   Pass 1: find max value
//   Pass 2: compute exp(x - max), sum, then normalize
// One threadgroup per row.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kSoftmaxSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     N      [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint row = tgid;
    device const half* in_row  = input  + row * N;
    device half*       out_row = output + row * N;

    // Pass 1: find max using simd+threadgroup reduction
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, float(in_row[i]));
    }
    local_max = simd_max(local_max);

    threadgroup float shared_max[32];
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) {
        shared_max[simd_idx] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_simdgroups = (tg_size + 31) / 32;
        float val = (simd_lane < n_simdgroups) ? shared_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        shared_max[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = shared_max[0];

    // Pass 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = exp(float(in_row[i]) - row_max);
        out_row[i] = half(val);
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_simdgroups = (tg_size + 31) / 32;
        float val = (simd_lane < n_simdgroups) ? shared_sum[simd_lane] : 0.0f;
        val = simd_sum(val);
        shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / shared_sum[0];

    // Pass 3: normalize
    for (uint i = tid; i < N; i += tg_size) {
        out_row[i] = half(float(out_row[i]) * inv_sum);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm: y = (x / rms(x)) * weight
//   rms(x) = sqrt(mean(x^2) + eps)
//
// One threadgroup per row. Simdgroup + threadgroup reduction for sum(x^2).
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kRMSNormSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const half*  input  [[buffer(0)]],
    device const half*  weight [[buffer(1)]],
    device half*        output [[buffer(2)]],
    constant uint&      N      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint row = tgid;
    device const half* in_row  = input  + row * N;
    device half*       out_row = output + row * N;

    // Sum of squares
    float local_ss = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float v = float(in_row[i]);
        local_ss += v * v;
    }
    local_ss = simd_sum(local_ss);

    threadgroup float shared_ss[32];
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) {
        shared_ss[simd_idx] = local_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_simdgroups = (tg_size + 31) / 32;
        float val = (simd_lane < n_simdgroups) ? shared_ss[simd_lane] : 0.0f;
        val = simd_sum(val);
        shared_ss[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_ss[0] / float(N) + eps);

    // Normalize and scale by weight
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(in_row[i]) * rms_inv * float(weight[i]);
        out_row[i] = half(val);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// SiLU activation: y = x / (1 + exp(-x))
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kSiluSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void silu(device const half* input [[buffer(0)]],
                 device half* output [[buffer(1)]],
                 constant uint& N [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    float x = float(input[tid]);
    output[tid] = half(x / (1.0f + exp(-x)));
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Elementwise multiply: output[i] = a[i] * b[i]
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kElementwiseMulSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    output[tid] = a[tid] * b[tid];
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Elementwise add: output[i] = a[i] + b[i]
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kElementwiseAddSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    output[tid] = a[tid] + b[tid];
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Embedding lookup: fetch rows from embedding table by token ID
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kEmbeddingLookupSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void embedding_lookup(device const half* table [[buffer(0)]],
                             device const uint* token_ids [[buffer(1)]],
                             device half* output [[buffer(2)]],
                             constant uint& embed_dim [[buffer(3)]],
                             constant uint& n_tokens [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
    uint tok_idx = tid / embed_dim;
    uint dim_idx = tid % embed_dim;
    if (tok_idx >= n_tokens) return;
    uint token_id = token_ids[tok_idx];
    output[tid] = table[token_id * embed_dim + dim_idx];
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 dequantization
//
// Q8_0 block layout (34 bytes per 32 elements):
//   [f16 scale (2B)] [32 × int8 quantized values (32B)]
//   Value = scale * quant[i]
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kDequantQ8_0Source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void dequantize_q8_0(device const uint8_t* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint& n_blocks [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) return;
    const uint block_size = 34;
    device const uint8_t* block = input + tid * block_size;
    half scale = *reinterpret_cast<device const half*>(block);
    device const int8_t* quants = reinterpret_cast<device const int8_t*>(block + 2);
    for (uint i = 0; i < 32; i++) {
        output[tid * 32 + i] = scale * half(float(quants[i]));
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Fused Q8_0 matrix-vector multiply
//
// Directly computes matvec from Q8_0-quantized matrix without materializing
// the dequantized matrix. Each threadgroup handles one row (M dimension).
// 256 threads per threadgroup, 8 simdgroups.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatvecQ8_0Source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matvec_q8_0(device const uint8_t* matrix [[buffer(0)]],
                        device const half* vec [[buffer(1)]],
                        device half* output [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid_in_tg [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_id [[simdgroup_index_in_threadgroup]]) {
    uint row = tgid;
    if (row >= M) return;
    const uint block_size_q8 = 34;
    const uint QK = 32;
    uint n_blocks_per_row = K / QK;
    device const uint8_t* row_data = matrix + row * n_blocks_per_row * block_size_q8;
    float sum = 0.0f;
    for (uint b = tid_in_tg; b < n_blocks_per_row; b += 256) {
        device const uint8_t* block = row_data + b * block_size_q8;
        half scale = *reinterpret_cast<device const half*>(block);
        device const int8_t* quants = reinterpret_cast<device const int8_t*>(block + 2);
        float block_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < QK; j++) {
            block_sum += float(quants[j]) * float(vec[b * QK + j]);
        }
        sum += float(scale) * block_sum;
    }
    sum = simd_sum(sum);
    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_in_tg == 0) {
        float total = 0.0f;
        uint n_simdgroups = min(256u / 32u, (n_blocks_per_row + 255u) / 256u + 1u);
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; s++) total += partial[s];
        output[row] = half(total);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Rotary Position Embedding (RoPE) — in-place on half-precision vectors
//
// x: n_heads * head_dim contiguous half values.
// Each thread handles one (cos, sin) rotation for one head.
// Grid: {n_heads * (head_dim / 2), 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kRoPESource = R"(
#include <metal_stdlib>
using namespace metal;

// NeoX-style (halved) RoPE: pairs are (x[i], x[i + head_dim/2])
kernel void rope(device half* x [[buffer(0)]],
                 constant uint& head_dim [[buffer(1)]],
                 constant uint& n_heads [[buffer(2)]],
                 constant uint& position [[buffer(3)]],
                 constant float& theta_base [[buffer(4)]],
                 uint tid [[thread_position_in_grid]]) {
    uint half_dim = head_dim / 2;
    uint head_idx = tid / half_dim;
    uint pair_idx = tid % half_dim;
    if (head_idx >= n_heads) return;

    float freq = 1.0f / pow(theta_base, float(2 * pair_idx) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint idx_re = head_idx * head_dim + pair_idx;
    uint idx_im = idx_re + half_dim;
    float re = float(x[idx_re]);
    float im = float(x[idx_im]);
    x[idx_re] = half(re * cos_val - im * sin_val);
    x[idx_im] = half(im * cos_val + re * sin_val);
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// MoE Gate — top-K expert selection with softmax normalization
//
// Single-thread kernel: finds top_k experts from gate_logits, applies softmax.
// Grid: {1, 1, 1}, Group: {1, 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMoEGateSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void moe_gate(device const half* gate_logits [[buffer(0)]],
                     device uint* top_indices [[buffer(1)]],
                     device half* top_weights [[buffer(2)]],
                     constant uint& n_experts [[buffer(3)]],
                     constant uint& top_k [[buffer(4)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    float probs[256];
    float max_val = -INFINITY;
    for (uint e = 0; e < n_experts; e++) {
        probs[e] = float(gate_logits[e]);
        max_val = max(max_val, probs[e]);
    }
    float sum = 0.0f;
    for (uint e = 0; e < n_experts; e++) {
        probs[e] = exp(probs[e] - max_val);
        sum += probs[e];
    }
    for (uint e = 0; e < n_experts; e++)
        probs[e] /= sum;

    for (uint k = 0; k < top_k; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts; e++) {
            bool already = false;
            for (uint p = 0; p < k; p++)
                if (top_indices[p] == e) { already = true; break; }
            if (!already && probs[e] > best_val) {
                best_val = probs[e];
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
        top_weights[k] = half(best_val);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// MoE Gate Grouped — DeepSeek V3 style grouped routing
//
// sigmoid → group top-2 sum → top-N groups → mask → global top-K → renorm
// Supports optional router bias (exp_probs_b) for expert selection.
// Grid: {1, 1, 1}  (single-thread; 256 experts is trivial)
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMoEGateGroupedSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void moe_gate_grouped(
    device const half* gate_logits [[buffer(0)]],
    device uint* top_indices       [[buffer(1)]],
    device half* top_weights       [[buffer(2)]],
    device const half* router_bias [[buffer(3)]],
    constant uint&  n_experts      [[buffer(4)]],
    constant uint&  top_k          [[buffer(5)]],
    constant uint&  n_group        [[buffer(6)]],
    constant uint&  topk_group     [[buffer(7)]],
    constant uint&  has_bias       [[buffer(8)]],
    constant float& scaling_factor [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    // Step 1: sigmoid scores
    float scores[256];
    for (uint e = 0; e < n_experts; e++) {
        float x = float(gate_logits[e]);
        scores[e] = 1.0f / (1.0f + exp(-x));
    }

    // Step 2: group scoring — top-2 per group, sum
    uint epg = n_experts / n_group;
    float group_scores[32];
    for (uint g = 0; g < n_group; g++) {
        float s1 = -INFINITY, s2 = -INFINITY;
        for (uint i = 0; i < epg; i++) {
            float s = scores[g * epg + i];
            if (s > s1) { s2 = s1; s1 = s; }
            else if (s > s2) { s2 = s; }
        }
        group_scores[g] = s1 + s2;
    }

    // Step 3: select top-N groups
    bool group_mask[32];
    for (uint g = 0; g < 32; g++) group_mask[g] = false;
    for (uint t = 0; t < topk_group; t++) {
        float best = -INFINITY;
        uint bg = 0;
        for (uint g = 0; g < n_group; g++) {
            if (!group_mask[g] && group_scores[g] > best) {
                best = group_scores[g];
                bg = g;
            }
        }
        group_mask[bg] = true;
    }

    // Step 4: global top-K on selected groups (bias shifts selection, not weights)
    for (uint k = 0; k < top_k && k < 64; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts; e++) {
            if (!group_mask[e / epg]) continue;
            float val = scores[e];
            if (has_bias) val += float(router_bias[e]);
            bool already = false;
            for (uint p = 0; p < k; p++) {
                if (top_indices[p] == e) { already = true; break; }
            }
            if (!already && val > best_val) {
                best_val = val;
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
    }

    // Step 5: renormalize — original sigmoid scores × scaling_factor
    float wsum = 0.0f;
    for (uint k = 0; k < top_k; k++)
        wsum += scores[top_indices[k]];
    for (uint k = 0; k < top_k; k++)
        top_weights[k] = half(scores[top_indices[k]] / wsum * scaling_factor);
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// MoE Reduce — weighted sum of K expert outputs
//
// output[i] = sum_k( expert_outputs[k * dim + i] * weights[k] )
// Grid: {dim, 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMoEReduceSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void moe_reduce(device const half* expert_outputs [[buffer(0)]],
                       device const half* weights [[buffer(1)]],
                       device half* output [[buffer(2)]],
                       constant uint& dim [[buffer(3)]],
                       constant uint& K [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(expert_outputs[k * dim + tid]) * float(weights[k]);
    }
    output[tid] = half(sum);
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Q4_K dequantization
//
// Q4_K block layout (144 bytes per 256 elements):
//   [f16 d (2B)] [f16 dmin (2B)] [12B packed scales] [128B 4-bit quants]
//   4 sub-blocks × 64 elements each.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kDequantQ4_KSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void dequantize_q4_k(device const uint8_t* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint& n_blocks [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) return;
    const uint BLOCK_SIZE = 144;
    device const uint8_t* block = input + tid * BLOCK_SIZE;

    half d = *reinterpret_cast<device const half*>(block);
    half dmin = *reinterpret_cast<device const half*>(block + 2);
    device const uint8_t* scales_packed = block + 4;
    device const uint8_t* qs = block + 16;

    float sc[4], mn[4];
    for (int i = 0; i < 4; i++) {
        uint8_t sl = scales_packed[i];
        uint8_t ml = scales_packed[4 + i];
        uint8_t sh_byte = scales_packed[8 + i];
        sc[i] = float(d) * float((sl & 0x3F) | ((sh_byte & 0x03) << 6));
        mn[i] = float(dmin) * float((ml & 0x3F) | (((sh_byte >> 2) & 0x03) << 6));
    }

    for (int sb = 0; sb < 4; sb++) {
        for (int j = 0; j < 32; j++) {
            uint8_t byte = qs[sb * 32 + j];
            output[tid * 256 + sb * 64 + j * 2]     = half(sc[sb] * float(byte & 0xF) - mn[sb]);
            output[tid * 256 + sb * 64 + j * 2 + 1] = half(sc[sb] * float(byte >> 4) - mn[sb]);
        }
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Fused Q4_K matrix-vector multiply
//
// Directly computes matvec from Q4_K-quantized matrix. Each threadgroup
// handles one row (M dimension). 256 threads per threadgroup, 8 simdgroups.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatvecQ4_KSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matvec_q4_k(device const uint8_t* matrix [[buffer(0)]],
                        device const half* vec [[buffer(1)]],
                        device half* output [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid_in_tg [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_id [[simdgroup_index_in_threadgroup]]) {
    uint row = tgid;
    if (row >= M) return;

    const uint BLOCK_SIZE = 144;
    const uint QK = 256;
    uint n_blocks_per_row = K / QK;
    device const uint8_t* row_data = matrix + row * n_blocks_per_row * BLOCK_SIZE;

    float sum = 0.0f;
    for (uint b = tid_in_tg; b < n_blocks_per_row; b += 256) {
        device const uint8_t* block = row_data + b * BLOCK_SIZE;
        half d = *reinterpret_cast<device const half*>(block);
        half dmin = *reinterpret_cast<device const half*>(block + 2);
        device const uint8_t* scales_packed = block + 4;
        device const uint8_t* qs = block + 16;

        float sc[4], mn[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            uint8_t sl = scales_packed[i];
            uint8_t ml = scales_packed[4 + i];
            uint8_t sh_byte = scales_packed[8 + i];
            sc[i] = float(d) * float((sl & 0x3F) | ((sh_byte & 0x03) << 6));
            mn[i] = float(dmin) * float((ml & 0x3F) | (((sh_byte >> 2) & 0x03) << 6));
        }

        float block_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int sb = 0; sb < 4; sb++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 32; j++) {
                uint8_t byte = qs[sb * 32 + j];
                uint base_idx = b * QK + sb * 64 + j * 2;
                float v0 = sc[sb] * float(byte & 0xF) - mn[sb];
                float v1 = sc[sb] * float(byte >> 4) - mn[sb];
                block_sum += v0 * float(vec[base_idx]) + v1 * float(vec[base_idx + 1]);
            }
        }
        sum += block_sum;
    }

    sum = simd_sum(sum);
    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_in_tg == 0) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; s++) total += partial[s];
        output[row] = half(total);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Fused Decode Attention (single-token decode, one threadgroup per Q head)
//
// Performs QK^T scoring, numerically stable softmax, and V weighted sum
// entirely on GPU. Supports GQA/MQA via heads_per_kv grouping.
// Grid: {n_heads * 256, 1, 1}, Group: {256, 1, 1}
// KV cache layout: token-major, each token = n_kv_heads * head_dim halfs.
// Max supported seq_len = 8160 (threadgroup memory bound).
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kAttentionDecodeSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void attention_decode(
    device const half* Q        [[buffer(0)]],
    device const half* K_cache  [[buffer(1)]],
    device const half* V_cache  [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    constant uint&     n_heads     [[buffer(4)]],
    constant uint&     n_kv_heads  [[buffer(5)]],
    constant uint&     head_dim    [[buffer(6)]],
    constant uint&     seq_len     [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head = tgid;
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    device const half* q_ptr = Q + head * head_dim;

    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float scores[8160];
    threadgroup float reduce_buf[32];

    // Phase 1: QK^T dot products
    for (uint s = tid; s < seq_len; s += tg_size) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Softmax — find max
    float local_max = -INFINITY;
    for (uint s = tid; s < seq_len; s += tg_size) {
        local_max = max(local_max, scores[s]);
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : -INFINITY;
        val = simd_max(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduce_buf[0];

    // Phase 2b: exp and sum
    float local_sum = 0.0f;
    for (uint s = tid; s < seq_len; s += tg_size) {
        float val = exp(scores[s] - global_max);
        scores[s] = val;
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : 0.0f;
        val = simd_sum(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce_buf[0];

    // Phase 3: V weighted sum — register accumulation + 2-barrier bulk reduction
    float partial_out[128];
    for (uint d = 0; d < head_dim; d++) partial_out[d] = 0.0f;

    for (uint s = tid; s < seq_len; s += tg_size) {
        float w = scores[s] * inv_sum;
        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            partial_out[d] += w * float(v_ptr[d]);
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        partial_out[d] = simd_sum(partial_out[d]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        for (uint d = 0; d < head_dim; d++) {
            scores[simd_idx * head_dim + d] = partial_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++) {
            acc += scores[sg * head_dim + tid];
        }
        output[head * head_dim + tid] = half(acc);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Batched Prefill Attention — causal mask + GQA
//
// One threadgroup per (token_idx, head) pair, 256 threads.
// Grid: { N * n_heads * 256, 1, 1 }, Group: { 256, 1, 1 }
// Processes Q_batch[N, n_heads, head_dim] against K/V cache with causal mask.
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kPrefillAttentionSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void prefill_attention(
    device const half* Q_batch    [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device half*       O_batch    [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& start_position [[buffer(8)]],
    constant uint& N              [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint token_idx = tgid / n_heads;
    uint head = tgid % n_heads;
    if (token_idx >= N) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    uint q_offset = token_idx * (n_heads * head_dim) + head * head_dim;
    device const half* q_ptr = Q_batch + q_offset;

    uint causal_len = start_position + token_idx + 1;

    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float scores[8160];
    threadgroup float reduce_buf[32];

    for (uint s = tid; s < causal_len; s += tg_size) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint s = tid; s < causal_len; s += tg_size) {
        local_max = max(local_max, scores[s]);
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : -INFINITY;
        val = simd_max(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < causal_len; s += tg_size) {
        float val = exp(scores[s] - global_max);
        scores[s] = val;
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : 0.0f;
        val = simd_sum(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce_buf[0];

    float partial_out[128];
    for (uint d = 0; d < head_dim; d++) partial_out[d] = 0.0f;

    for (uint s = tid; s < causal_len; s += tg_size) {
        float w = scores[s] * inv_sum;
        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            partial_out[d] += w * float(v_ptr[d]);
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        partial_out[d] = simd_sum(partial_out[d]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        for (uint d = 0; d < head_dim; d++) {
            scores[simd_idx * head_dim + d] = partial_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++) {
            acc += scores[sg * head_dim + tid];
        }
        O_batch[q_offset + tid] = half(acc);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Matrix-Matrix Multiplication — FP16
//
// C = B × A^T   (B: N×K input batch, A: M×K weights, C: N×M output batch)
// All row-major. Each thread computes one output element.
// Grid: {ceil(M,16)*16, ceil(N,16)*16, 1}, Group: {16, 16, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatmulF16Source = R"(
#include <metal_stdlib>
using namespace metal;

// Tiled GEMM: C = B × Aᵀ, simdgroup 8×8 MMA.
// 64×32 output tile (M×N), 128 threads (4 simdgroups), K-tile=32.
kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& K_dim [[buffer(4)]],
    constant uint& N    [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[64 * 33];
    threadgroup half sb[32 * 33];
    threadgroup float tc[64 * 32];

    const uint M_base = tgpig.x * 64;
    const uint N_base = tgpig.y * 32;
    const ushort sg_m = sgitg & 1;
    const ushort sg_n = sgitg >> 1;

    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    for (uint k = 0; k < K_dim; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort idx = tiitg; idx < 2048; idx += 128) {
            ushort r = idx >> 5, c = idx & 31;
            uint am = M_base + r, ak = k + c;
            sa[r * 33 + c] = (am < M && ak < K_dim) ? A[(uint64_t)am * K_dim + ak] : half(0);
        }
        for (ushort idx = tiitg; idx < 1024; idx += 128) {
            ushort r = idx >> 5, c = idx & 31;
            uint bn = N_base + r, bk = k + c;
            sb[r * 33 + c] = (bn < N && bk < K_dim) ? B[(uint64_t)bn * K_dim + bk] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma[4], mb[2];
        for (ushort kk = 0; kk < 32; kk += 8) {
            for (ushort i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa, 33, ulong2(kk, sg_m * 32 + i * 8), true);
            simdgroup_load(mb[0], sb, 33, ulong2(kk, sg_n * 16),     false);
            simdgroup_load(mb[1], sb, 33, ulong2(kk, sg_n * 16 + 8), false);
            for (ushort i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
        }
    }

    for (ushort i = 0; i < 8; i++) {
        ushort m_off = sg_m * 32 + (i % 4) * 8;
        ushort n_off = sg_n * 16 + (i / 4) * 8;
        simdgroup_store(mc[i], tc + n_off * 64 + m_off, 64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort idx = tiitg; idx < 2048; idx += 128) {
        ushort m = idx & 63, n = idx >> 6;
        uint gm = M_base + m, gn = N_base + n;
        if (gm < M && gn < N)
            C[(uint64_t)gn * M + gm] = half(tc[n * 64 + m]);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Matrix-Matrix Multiplication — Q4_0 weights
//
// C = B × A^T   (A: Q4_0 M×K, B: f16 N×K, C: f16 N×M)
// Q4_0 block: 18 bytes = 2B f16 scale + 16B nibbles (32 values).
// Each value = (nibble - 8) * scale.
// Grid: {ceil(M,16)*16, ceil(N,16)*16, 1}, Group: {16, 16, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kMatmulQ4_0Source = R"(
#include <metal_stdlib>
using namespace metal;

// Tiled GEMM for Q4_0: C = B × Aᵀ, dequantize to threadgroup.
// 64×32 output tile (M×N), 128 threads (4 simdgroups), K-tile=32.
kernel void matmul_q4_0(
    device const uint8_t* A [[buffer(0)]],
    device const half*    B [[buffer(1)]],
    device half*          C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K_dim  [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[64 * 33];
    threadgroup half sb[32 * 33];
    threadgroup float tc[64 * 32];

    const uint M_base = tgpig.x * 64;
    const uint N_base = tgpig.y * 32;
    const ushort sg_m = sgitg & 1;
    const ushort sg_n = sgitg >> 1;

    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    constant uint QK = 32;
    constant uint BLK = 18;
    const uint nb = K_dim / QK;

    for (uint k = 0; k < K_dim; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint blk_col = k / QK;
        ushort row_id = tiitg >> 1;
        ushort half_id = tiitg & 1;
        uint am = M_base + row_id;
        if (am < M && k < K_dim) {
            device const uint8_t* blk = A + (uint64_t)am * nb * BLK
                                           + (uint64_t)blk_col * BLK;
            half scale = *reinterpret_cast<device const half*>(blk);
            device const uint8_t* qs = blk + 2;
            for (ushort j = 0; j < 8; j++) {
                uint8_t byte = qs[half_id * 8 + j];
                sa[row_id * 33 + half_id * 8 + j]      = scale * half((int)(byte & 0x0F) - 8);
                sa[row_id * 33 + half_id * 8 + j + 16]   = scale * half((int)(byte >> 4) - 8);
            }
        } else {
            for (ushort j = 0; j < 16; j++)
                sa[row_id * 33 + half_id * 16 + j] = half(0);
        }

        for (ushort idx = tiitg; idx < 1024; idx += 128) {
            ushort r = idx >> 5, c = idx & 31;
            uint bn = N_base + r, bk = k + c;
            sb[r * 33 + c] = (bn < N && bk < K_dim) ? B[(uint64_t)bn * K_dim + bk] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma[4], mb[2];
        for (ushort kk = 0; kk < 32; kk += 8) {
            for (ushort i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa, 33, ulong2(kk, sg_m * 32 + i * 8), true);
            simdgroup_load(mb[0], sb, 33, ulong2(kk, sg_n * 16),     false);
            simdgroup_load(mb[1], sb, 33, ulong2(kk, sg_n * 16 + 8), false);
            for (ushort i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
        }
    }

    for (ushort i = 0; i < 8; i++) {
        ushort m_off = sg_m * 32 + (i % 4) * 8;
        ushort n_off = sg_n * 16 + (i / 4) * 8;
        simdgroup_store(mc[i], tc + n_off * 64 + m_off, 64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort idx = tiitg; idx < 2048; idx += 128) {
        ushort m = idx & 63, n = idx >> 6;
        uint gm = M_base + m, gn = N_base + n;
        if (gm < M && gn < N)
            C[(uint64_t)gn * M + gm] = half(tc[n * 64 + m]);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Batch RoPE — apply RoPE to N tokens with sequential positions
//
// NeoX-style halved: pair = (x[i], x[i + half_dim])
// position[tok] = start_pos + tok
// Input layout: N_tok × n_heads × head_dim (contiguous)
// Grid: {n_tokens * n_heads * (head_dim/2), 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kBatchRopeSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void batch_rope(
    device half* x          [[buffer(0)]],
    constant uint& head_dim    [[buffer(1)]],
    constant uint& n_heads     [[buffer(2)]],
    constant uint& start_pos   [[buffer(3)]],
    constant float& theta_base [[buffer(4)]],
    constant uint& n_tokens    [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint pairs_per_token = n_heads * half_dim;
    uint tok_idx = tid / pairs_per_token;
    uint rem = tid % pairs_per_token;
    uint head_idx = rem / half_dim;
    uint pair_idx = rem % half_dim;

    if (tok_idx >= n_tokens) return;

    uint position = start_pos + tok_idx;
    float freq = 1.0 / pow(theta_base, float(2 * pair_idx) / float(head_dim));
    float angle = float(position) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint base = tok_idx * n_heads * head_dim + head_idx * head_dim;
    uint idx_re = base + pair_idx;
    uint idx_im = idx_re + half_dim;

    float re = float(x[idx_re]);
    float im = float(x[idx_im]);
    x[idx_re] = half(re * cos_a - im * sin_a);
    x[idx_im] = half(im * cos_a + re * sin_a);
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Bias Broadcast — add 1D bias to each row of a 2D batch buffer (in-place)
//
// batch[gid] += bias[gid % dim]
// Grid: {N_tokens * dim, 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kBiasBroadcastSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bias_broadcast(
    device half* batch        [[buffer(0)]],
    device const half* bias   [[buffer(1)]],
    constant uint& dim        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint col = gid % dim;
    batch[gid] = batch[gid] + bias[col];
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Flash Prefill Attention — online softmax, tiled KV processing
//
// Replaces prefill_attention with O(1) threadgroup memory per KV position.
// Processes KV in tiles of 256, maintaining running max + sum for exact
// softmax without materializing the full scores array.
// Supports arbitrary sequence lengths (no 8160-token limit).
// Grid: { N * n_heads * 256, 1, 1 }, Group: { 256, 1, 1 }
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kFlashPrefillAttentionSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void flash_prefill_attention(
    device const half* Q_batch    [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device half*       O_batch    [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& start_position [[buffer(8)]],
    constant uint& N              [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint token_idx = tgid / n_heads;
    uint head = tgid % n_heads;
    if (token_idx >= N) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    uint q_offset = token_idx * (n_heads * head_dim) + head * head_dim;
    device const half* q_ptr = Q_batch + q_offset;

    uint causal_len = start_position + token_idx + 1;

    uint sg_id = tid / 32;
    uint lane  = tid % 32;
    uint n_sg  = tg_size / 32;
    uint dim_base = lane * 4;

    float q_reg[4];
    for (uint i = 0; i < 4; i++)
        q_reg[i] = (dim_base + i < head_dim) ? float(q_ptr[dim_base + i]) : 0.0f;

    const uint TILE_KV = 256;
    threadgroup float tile_mem[1024];
    threadgroup float reduce_buf[32];

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tile_start = 0; tile_start < causal_len; tile_start += TILE_KV) {
        uint tile_end = min(tile_start + TILE_KV, causal_len);
        uint tile_len = tile_end - tile_start;

        for (uint s = sg_id; s < tile_len; s += n_sg) {
            uint kv_pos = tile_start + s;
            device const half* k_ptr = K_cache + kv_pos * kv_stride + kv_head * head_dim;
            float partial = 0.0f;
            for (uint i = 0; i < 4; i++)
                partial += q_reg[i] * ((dim_base + i < head_dim) ? float(k_ptr[dim_base + i]) : 0.0f);
            float dot = simd_sum(partial) * scale;
            if (lane == 0) tile_mem[s] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -INFINITY;
        for (uint s = tid; s < tile_len; s += tg_size) {
            local_max = max(local_max, tile_mem[s]);
        }
        local_max = simd_max(local_max);
        if (lane == 0) reduce_buf[sg_id] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float val = (lane < n_sg) ? reduce_buf[lane] : -INFINITY;
            val = simd_max(val);
            reduce_buf[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = reduce_buf[0];

        float new_max = max(running_max, tile_max);
        float exp_old = exp(running_max - new_max);

        float local_sum = 0.0f;
        for (uint s = tid; s < tile_len; s += tg_size) {
            float val = exp(tile_mem[s] - new_max);
            tile_mem[s] = val;
            local_sum += val;
        }
        local_sum = simd_sum(local_sum);
        if (lane == 0) reduce_buf[sg_id] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float val = (lane < n_sg) ? reduce_buf[lane] : 0.0f;
            val = simd_sum(val);
            reduce_buf[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_sum = reduce_buf[0];

        float rescale = (running_sum > 0.0f) ? exp_old : 0.0f;
        for (uint i = 0; i < 4; i++) acc[i] *= rescale;

        for (uint s = sg_id; s < tile_len; s += n_sg) {
            float w = tile_mem[s];
            uint kv_pos = tile_start + s;
            device const half* v_ptr = V_cache + kv_pos * kv_stride + kv_head * head_dim;
            for (uint i = 0; i < 4; i++)
                acc[i] += w * ((dim_base + i < head_dim) ? float(v_ptr[dim_base + i]) : 0.0f);
        }

        running_max = new_max;
        running_sum = running_sum * exp_old + tile_sum;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint i = 0; i < 4; i++) acc[i] *= inv_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; i++) {
        if (dim_base + i < head_dim)
            tile_mem[sg_id * head_dim + dim_base + i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float result = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++)
            result += tile_mem[sg * head_dim + tid];
        O_batch[q_offset + tid] = half(result);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Flash Attention Decode — online softmax, tiled KV, no scores[8160] limit
//
// Single-query variant: simdgroup-parallel Q·K dot products + online softmax.
// 8 simdgroups (256 threads), each processes a subset of seq positions.
// 32 lanes per simdgroup × 4 dims/lane = 128 dims collaborative dot product.
// Grid: { n_heads * 256, 1, 1 }, Group: { 256, 1, 1 }
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kFlashAttentionDecodeSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void flash_attention_decode(
    device const half* Q        [[buffer(0)]],
    device const half* K_cache  [[buffer(1)]],
    device const half* V_cache  [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    constant uint&     n_heads     [[buffer(4)]],
    constant uint&     n_kv_heads  [[buffer(5)]],
    constant uint&     head_dim    [[buffer(6)]],
    constant uint&     seq_len     [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head = tgid;
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    device const half* q_ptr = Q + head * head_dim;

    uint sg_id = tid / 32;
    uint lane  = tid % 32;
    uint n_sg  = tg_size / 32;
    if (n_sg > 8) return;  // 守护：threadgroup 太大则跳过
    uint dim_base = lane * 4;

    float q_reg[4];
    for (uint i = 0; i < 4; i++)
        q_reg[i] = (dim_base + i < head_dim) ? float(q_ptr[dim_base + i]) : 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint s = sg_id; s < seq_len; s += n_sg) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;

        float partial = 0.0f;
        for (uint i = 0; i < 4; i++)
            partial += q_reg[i] * ((dim_base + i < head_dim) ? float(k_ptr[dim_base + i]) : 0.0f);
        float dot = simd_sum(partial) * scale;

        float new_max = max(running_max, dot);
        float exp_old = exp(running_max - new_max);
        float exp_new = exp(dot - new_max);
        float rescale = (running_sum > 0.0f) ? exp_old : 0.0f;

        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint i = 0; i < 4; i++) {
            float v_val = (dim_base + i < head_dim) ? float(v_ptr[dim_base + i]) : 0.0f;
            acc[i] = acc[i] * rescale + exp_new * v_val;
        }

        running_max = new_max;
        running_sum = running_sum * exp_old + exp_new;
    }

    threadgroup float sg_max[8];
    threadgroup float sg_sum[8];
    threadgroup float sg_acc[8 * 128];
    threadgroup float tg_global_max;
    threadgroup float tg_inv_sum;

    if (lane == 0) {
        sg_max[sg_id] = running_max;
        sg_sum[sg_id] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gmax = -INFINITY;
        for (uint i = 0; i < n_sg; i++)
            gmax = max(gmax, sg_max[i]);
        tg_global_max = gmax;
        float gsum = 0.0f;
        for (uint i = 0; i < n_sg; i++) {
            if (sg_sum[i] > 0.0f)
                gsum += sg_sum[i] * exp(sg_max[i] - gmax);
        }
        tg_inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_global_max;
    float inv_sum = tg_inv_sum;
    float my_rescale = (running_sum > 0.0f) ? exp(running_max - global_max) : 0.0f;

    for (uint i = 0; i < 4; i++)
        sg_acc[sg_id * 128 + dim_base + i] = acc[i] * my_rescale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float result = 0.0f;
        for (uint i = 0; i < n_sg; i++)
            result += sg_acc[i * 128 + tid];
        output[head * head_dim + tid] = half(result * inv_sum);
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Scatter KV — GPU-side copy from temp K/V buffer into persistent KV cache
//
// Copies `count` half elements from src to dst+offset.
// Replaces CPU memcpy for GPU KV buffer updates.
// Grid: {ceil(count/256)*256, 1, 1}, Group: {256, 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kScatterKvSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void scatter_kv(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& offset  [[buffer(2)]],
    constant uint& count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        dst[offset + tid] = src[tid];
    }
}
)";

// ═══════════════════════════════════════════════════════════════════════════
// Argmax F16 — single-threadgroup reduce to find index of max element
//
// Input: half[N], Output: uint[1] (index of max element)
// Single threadgroup, 1024 threads. Each thread scans ceil(N/1024) elements.
// SIMD reduce → threadgroup reduce via shared memory → thread 0 writes result.
// Grid: {1024, 1, 1}, Group: {1024, 1, 1}
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr const char* kArgmaxF16Source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f16(
    device const half* input [[buffer(0)]],
    device uint* result      [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = tid; i < N; i += tg_size) {
        float v = float(input[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    local_max = simd_max(local_max);
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    // Broadcast winning index within simdgroup
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(local_max, offset);
        uint other_idx  = simd_shuffle_down(local_idx, offset);
        if (other_val > local_max) { local_max = other_val; local_idx = other_idx; }
    }

    threadgroup float shared_val[32];
    threadgroup uint  shared_idx[32];
    uint n_sg = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared_val[simd_idx] = local_max;
        shared_idx[simd_idx] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? shared_val[simd_lane] : -INFINITY;
        uint  idx = (simd_lane < n_sg) ? shared_idx[simd_lane] : 0;

        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(val, offset);
            uint  other_idx = simd_shuffle_down(idx, offset);
            if (other_val > val) { val = other_val; idx = other_idx; }
        }

        if (simd_lane == 0) {
            result[0] = idx;
        }
    }
}
)";

// Convenience: all kernel sources concatenated for single-library compilation
inline constexpr const char* kAllKernelsSource = R"(
#include <metal_stdlib>
using namespace metal;

// ─── Q4_0 dequantization ───

constant constexpr uint QK4_0 = 32;
constant constexpr uint BLOCK_SIZE_Q4_0 = 18;

kernel void dequantize_q4_0(
    device const uint8_t* input  [[buffer(0)]],
    device half*          output [[buffer(1)]],
    constant uint&        n_blocks [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_blocks) return;

    device const uint8_t* block = input + tid * BLOCK_SIZE_Q4_0;
    half scale = *reinterpret_cast<device const half*>(block);
    device const uint8_t* data = block + 2;
    device half* out = output + tid * QK4_0;

    for (uint j = 0; j < 16; ++j) {
        uint8_t byte = data[j];
        int lo = (int)(byte & 0x0F) - 8;
        int hi = (int)(byte >> 4)    - 8;
        out[j]      = scale * (half)lo;
        out[j + 16] = scale * (half)hi;
    }
}

// ─── Half-precision matvec (one threadgroup per row) ───

kernel void matvec_f16(
    device const half* matrix [[buffer(0)]],
    device const half* vec    [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     M      [[buffer(3)]],
    constant uint&     K      [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid;
    if (row >= M) return;

    device const half* row_ptr = matrix + (uint64_t)row * K;

    float acc = 0.0f;
    for (uint col = tid; col < K; col += 32) {
        acc += float(row_ptr[col]) * float(vec[col]);
    }
    acc = simd_sum(acc);

    if (lane == 0) {
        output[row] = half(acc);
    }
}

// ─── Fused Q4_0 matvec (8 rows/tg, pre-scaled vec, raw uint16 accumulate) ───

kernel void matvec_q4_0(
    device const uint8_t* matrix_q4 [[buffer(0)]],
    device const half*    vec       [[buffer(1)]],
    device half*          output    [[buffer(2)]],
    constant uint&        M         [[buffer(3)]],
    constant uint&        K         [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint NR0 = 4;
    const uint NSG = 2;
    const uint NQ  = 16;

    const uint nb = K / QK4_0;
    const uint r0 = (tgid * NSG + sgitg) * NR0;

    if (r0 >= M) return;

    const short ix = short(tiisg / 2);
    const short il = short(tiisg % 2) * 8;

    device const uint8_t* ax[NR0];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < NR0; i++) {
        uint row = min(r0 + i, M - 1);
        ax[i] = matrix_q4 + (uint64_t)row * nb * BLOCK_SIZE_Q4_0;
    }

    float sumf[NR0] = {};

    device const half* yb = vec + uint(ix) * QK4_0 + uint(il);

    for (uint ib = uint(ix); ib < nb; ib += NQ) {
        float sumy = 0.f;
        float yl[16];

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i += 2) {
            float v0 = float(yb[i]);
            float v1 = float(yb[i + 1]);
            float v2 = float(yb[i + 16]);
            float v3 = float(yb[i + 17]);

            sumy += v0 + v1 + v2 + v3;

            yl[i + 0] = v0;
            yl[i + 1] = v1 / 256.f;
            yl[i + 8] = v2 / 16.f;
            yl[i + 9] = v3 / 4096.f;
        }

        #pragma clang loop unroll(full)
        for (uint row = 0; row < NR0; row++) {
            device const uint8_t* blk = ax[row] + (uint64_t)ib * BLOCK_SIZE_Q4_0;
            float d = float(*reinterpret_cast<device const half*>(blk));
            device const uint16_t* qs =
                reinterpret_cast<device const uint16_t*>(blk + 2) + il / 2;

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                acc0 += yl[i + 0] * float(qs[i / 2] & 0x000Fu);
                acc1 += yl[i + 1] * float(qs[i / 2] & 0x0F00u);
                acc2 += yl[i + 8] * float(qs[i / 2] & 0x00F0u);
                acc3 += yl[i + 9] * float(qs[i / 2] & 0xF000u);
            }

            sumf[row] += d * (sumy * -8.f + acc0 + acc1 + acc2 + acc3);
        }

        yb += QK4_0 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < M) {
            output[r0 + row] = half(tot);
        }
    }
}

// ─── Softmax (numerically stable, one row per threadgroup) ───

kernel void softmax(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     N      [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint row = tgid;
    device const half* in_row  = input  + row * N;
    device half*       out_row = output + row * N;

    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, float(in_row[i]));
    }
    local_max = simd_max(local_max);

    threadgroup float shared_max[32];
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) shared_max[simd_idx] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float val = (simd_lane < n_sg) ? shared_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        shared_max[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_max[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = exp(float(in_row[i]) - row_max);
        out_row[i] = half(val);
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    if (simd_lane == 0) shared_sum[simd_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float val = (simd_lane < n_sg) ? shared_sum[simd_lane] : 0.0f;
        val = simd_sum(val);
        shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / shared_sum[0];
    for (uint i = tid; i < N; i += tg_size) {
        out_row[i] = half(float(out_row[i]) * inv_sum);
    }
}

// ─── RMSNorm (one row per threadgroup) ───

kernel void rms_norm(
    device const half*  input  [[buffer(0)]],
    device const half*  weight [[buffer(1)]],
    device half*        output [[buffer(2)]],
    constant uint&      N      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint row = tgid;
    device const half* in_row  = input  + row * N;
    device half*       out_row = output + row * N;

    float local_ss = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float v = float(in_row[i]);
        local_ss += v * v;
    }
    local_ss = simd_sum(local_ss);

    threadgroup float shared_ss[32];
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) shared_ss[simd_idx] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float val = (simd_lane < n_sg) ? shared_ss[simd_lane] : 0.0f;
        val = simd_sum(val);
        shared_ss[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_ss[0] / float(N) + eps);

    for (uint i = tid; i < N; i += tg_size) {
        float val = float(in_row[i]) * rms_inv * float(weight[i]);
        out_row[i] = half(val);
    }
}

// ─── SiLU activation ───

kernel void silu(device const half* input [[buffer(0)]],
                 device half* output [[buffer(1)]],
                 constant uint& N [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    float x = float(input[tid]);
    output[tid] = half(x / (1.0f + exp(-x)));
}

// ─── Elementwise multiply ───

kernel void elementwise_mul(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    output[tid] = a[tid] * b[tid];
}

// ─── Elementwise add ───

kernel void elementwise_add(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= N) return;
    output[tid] = a[tid] + b[tid];
}

// ─── Embedding lookup ───

kernel void embedding_lookup(device const half* table [[buffer(0)]],
                             device const uint* token_ids [[buffer(1)]],
                             device half* output [[buffer(2)]],
                             constant uint& embed_dim [[buffer(3)]],
                             constant uint& n_tokens [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
    uint tok_idx = tid / embed_dim;
    uint dim_idx = tid % embed_dim;
    if (tok_idx >= n_tokens) return;
    uint token_id = token_ids[tok_idx];
    output[tid] = table[token_id * embed_dim + dim_idx];
}

// ─── Q8_0 dequantization ───

kernel void dequantize_q8_0(device const uint8_t* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint& n_blocks [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) return;
    const uint block_size = 34;
    device const uint8_t* block = input + tid * block_size;
    half scale = *reinterpret_cast<device const half*>(block);
    device const int8_t* quants = reinterpret_cast<device const int8_t*>(block + 2);
    for (uint i = 0; i < 32; i++) {
        output[tid * 32 + i] = scale * half(float(quants[i]));
    }
}

// ─── Fused Q8_0 matvec ───

kernel void matvec_q8_0(device const uint8_t* matrix [[buffer(0)]],
                        device const half* vec [[buffer(1)]],
                        device half* output [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid_in_tg [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_id [[simdgroup_index_in_threadgroup]]) {
    uint row = tgid;
    if (row >= M) return;
    const uint block_size_q8 = 34;
    const uint QK = 32;
    uint n_blocks_per_row = K / QK;
    device const uint8_t* row_data = matrix + row * n_blocks_per_row * block_size_q8;
    float sum = 0.0f;
    for (uint b = tid_in_tg; b < n_blocks_per_row; b += 256) {
        device const uint8_t* block = row_data + b * block_size_q8;
        half scale = *reinterpret_cast<device const half*>(block);
        device const int8_t* quants = reinterpret_cast<device const int8_t*>(block + 2);
        float block_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < QK; j++) {
            block_sum += float(quants[j]) * float(vec[b * QK + j]);
        }
        sum += float(scale) * block_sum;
    }
    sum = simd_sum(sum);
    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_in_tg == 0) {
        float total = 0.0f;
        uint n_simdgroups = min(256u / 32u, (n_blocks_per_row + 255u) / 256u + 1u);
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; s++) total += partial[s];
        output[row] = half(total);
    }
}

// ─── RoPE (rotary position embedding, in-place, NeoX-style halved) ───

kernel void rope(device half* x [[buffer(0)]],
                 constant uint& head_dim [[buffer(1)]],
                 constant uint& n_heads [[buffer(2)]],
                 constant uint& position [[buffer(3)]],
                 constant float& theta_base [[buffer(4)]],
                 uint tid [[thread_position_in_grid]]) {
    uint half_dim = head_dim / 2;
    uint head_idx = tid / half_dim;
    uint pair_idx = tid % half_dim;
    if (head_idx >= n_heads) return;

    float freq = 1.0f / pow(theta_base, float(2 * pair_idx) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint idx_re = head_idx * head_dim + pair_idx;
    uint idx_im = idx_re + half_dim;
    float re = float(x[idx_re]);
    float im = float(x[idx_im]);
    x[idx_re] = half(re * cos_val - im * sin_val);
    x[idx_im] = half(im * cos_val + re * sin_val);
}

// ─── MoE gate (top-K selection + softmax) ───

kernel void moe_gate(device const half* gate_logits [[buffer(0)]],
                     device uint* top_indices [[buffer(1)]],
                     device half* top_weights [[buffer(2)]],
                     constant uint& n_experts [[buffer(3)]],
                     constant uint& top_k [[buffer(4)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    float probs[256];
    float max_val = -INFINITY;
    for (uint e = 0; e < n_experts; e++) {
        probs[e] = float(gate_logits[e]);
        max_val = max(max_val, probs[e]);
    }
    float sum = 0.0f;
    for (uint e = 0; e < n_experts; e++) {
        probs[e] = exp(probs[e] - max_val);
        sum += probs[e];
    }
    for (uint e = 0; e < n_experts; e++)
        probs[e] /= sum;

    for (uint k = 0; k < top_k; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts; e++) {
            bool already = false;
            for (uint p = 0; p < k; p++)
                if (top_indices[p] == e) { already = true; break; }
            if (!already && probs[e] > best_val) {
                best_val = probs[e];
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
        top_weights[k] = half(best_val);
    }
}

// ─── MoE gate grouped (DeepSeek V3 style: sigmoid + group routing + renorm) ───

kernel void moe_gate_grouped(
    device const half* gate_logits [[buffer(0)]],
    device uint* top_indices       [[buffer(1)]],
    device half* top_weights       [[buffer(2)]],
    device const half* router_bias [[buffer(3)]],
    constant uint&  n_experts      [[buffer(4)]],
    constant uint&  top_k          [[buffer(5)]],
    constant uint&  n_group        [[buffer(6)]],
    constant uint&  topk_group     [[buffer(7)]],
    constant uint&  has_bias       [[buffer(8)]],
    constant float& scaling_factor [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    float scores[256];
    for (uint e = 0; e < n_experts; e++) {
        float x = float(gate_logits[e]);
        scores[e] = 1.0f / (1.0f + exp(-x));
    }

    uint epg = n_experts / n_group;
    float group_scores[32];
    for (uint g = 0; g < n_group; g++) {
        float s1 = -INFINITY, s2 = -INFINITY;
        for (uint i = 0; i < epg; i++) {
            float s = scores[g * epg + i];
            if (s > s1) { s2 = s1; s1 = s; }
            else if (s > s2) { s2 = s; }
        }
        group_scores[g] = s1 + s2;
    }

    bool group_mask[32];
    for (uint g = 0; g < 32; g++) group_mask[g] = false;
    for (uint t = 0; t < topk_group; t++) {
        float best = -INFINITY;
        uint bg = 0;
        for (uint g = 0; g < n_group; g++) {
            if (!group_mask[g] && group_scores[g] > best) {
                best = group_scores[g];
                bg = g;
            }
        }
        group_mask[bg] = true;
    }

    for (uint k = 0; k < top_k && k < 64; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts; e++) {
            if (!group_mask[e / epg]) continue;
            float val = scores[e];
            if (has_bias) val += float(router_bias[e]);
            bool already = false;
            for (uint p = 0; p < k; p++) {
                if (top_indices[p] == e) { already = true; break; }
            }
            if (!already && val > best_val) {
                best_val = val;
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
    }

    float wsum = 0.0f;
    for (uint k = 0; k < top_k; k++)
        wsum += scores[top_indices[k]];
    for (uint k = 0; k < top_k; k++)
        top_weights[k] = half(scores[top_indices[k]] / wsum * scaling_factor);
}

// ─── MoE reduce (weighted sum of expert outputs) ───

kernel void moe_reduce(device const half* expert_outputs [[buffer(0)]],
                       device const half* weights [[buffer(1)]],
                       device half* output [[buffer(2)]],
                       constant uint& dim [[buffer(3)]],
                       constant uint& K [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(expert_outputs[k * dim + tid]) * float(weights[k]);
    }
    output[tid] = half(sum);
}

// ─── Q4_K dequantization (144 bytes/block, 256 elements) ───

kernel void dequantize_q4_k(device const uint8_t* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint& n_blocks [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) return;
    const uint BLOCK_SIZE = 144;
    device const uint8_t* block = input + tid * BLOCK_SIZE;

    half d = *reinterpret_cast<device const half*>(block);
    half dmin = *reinterpret_cast<device const half*>(block + 2);
    device const uint8_t* scales_packed = block + 4;
    device const uint8_t* qs = block + 16;

    float sc[4], mn[4];
    for (int i = 0; i < 4; i++) {
        uint8_t sl = scales_packed[i];
        uint8_t ml = scales_packed[4 + i];
        uint8_t sh_byte = scales_packed[8 + i];
        sc[i] = float(d) * float((sl & 0x3F) | ((sh_byte & 0x03) << 6));
        mn[i] = float(dmin) * float((ml & 0x3F) | (((sh_byte >> 2) & 0x03) << 6));
    }

    for (int sb = 0; sb < 4; sb++) {
        for (int j = 0; j < 32; j++) {
            uint8_t byte = qs[sb * 32 + j];
            output[tid * 256 + sb * 64 + j * 2]     = half(sc[sb] * float(byte & 0xF) - mn[sb]);
            output[tid * 256 + sb * 64 + j * 2 + 1] = half(sc[sb] * float(byte >> 4) - mn[sb]);
        }
    }
}

// ─── Fused Q4_K matvec (one threadgroup per row, 256 threads) ───

kernel void matvec_q4_k(device const uint8_t* matrix [[buffer(0)]],
                        device const half* vec [[buffer(1)]],
                        device half* output [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid_in_tg [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_id [[simdgroup_index_in_threadgroup]]) {
    uint row = tgid;
    if (row >= M) return;

    const uint BLOCK_SIZE = 144;
    const uint QK = 256;
    uint n_blocks_per_row = K / QK;
    device const uint8_t* row_data = matrix + row * n_blocks_per_row * BLOCK_SIZE;

    float sum = 0.0f;
    for (uint b = tid_in_tg; b < n_blocks_per_row; b += 256) {
        device const uint8_t* block = row_data + b * BLOCK_SIZE;
        half d = *reinterpret_cast<device const half*>(block);
        half dmin = *reinterpret_cast<device const half*>(block + 2);
        device const uint8_t* scales_packed = block + 4;
        device const uint8_t* qs = block + 16;

        float sc[4], mn[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            uint8_t sl = scales_packed[i];
            uint8_t ml = scales_packed[4 + i];
            uint8_t sh_byte = scales_packed[8 + i];
            sc[i] = float(d) * float((sl & 0x3F) | ((sh_byte & 0x03) << 6));
            mn[i] = float(dmin) * float((ml & 0x3F) | (((sh_byte >> 2) & 0x03) << 6));
        }

        float block_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int sb = 0; sb < 4; sb++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 32; j++) {
                uint8_t byte = qs[sb * 32 + j];
                uint base_idx = b * QK + sb * 64 + j * 2;
                float v0 = sc[sb] * float(byte & 0xF) - mn[sb];
                float v1 = sc[sb] * float(byte >> 4) - mn[sb];
                block_sum += v0 * float(vec[base_idx]) + v1 * float(vec[base_idx + 1]);
            }
        }
        sum += block_sum;
    }

    sum = simd_sum(sum);
    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_in_tg == 0) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; s++) total += partial[s];
        output[row] = half(total);
    }
}

// ─── Fused decode attention (one threadgroup per Q head, 256 threads) ───

kernel void attention_decode(
    device const half* Q        [[buffer(0)]],
    device const half* K_cache  [[buffer(1)]],
    device const half* V_cache  [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    constant uint&     n_heads     [[buffer(4)]],
    constant uint&     n_kv_heads  [[buffer(5)]],
    constant uint&     head_dim    [[buffer(6)]],
    constant uint&     seq_len     [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head = tgid;
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    device const half* q_ptr = Q + head * head_dim;

    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float scores[8160];
    threadgroup float reduce_buf[32];

    for (uint s = tid; s < seq_len; s += tg_size) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint s = tid; s < seq_len; s += tg_size) {
        local_max = max(local_max, scores[s]);
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : -INFINITY;
        val = simd_max(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < seq_len; s += tg_size) {
        float val = exp(scores[s] - global_max);
        scores[s] = val;
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : 0.0f;
        val = simd_sum(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce_buf[0];

    float partial_out[128];
    for (uint d = 0; d < head_dim; d++) partial_out[d] = 0.0f;

    for (uint s = tid; s < seq_len; s += tg_size) {
        float w = scores[s] * inv_sum;
        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            partial_out[d] += w * float(v_ptr[d]);
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        partial_out[d] = simd_sum(partial_out[d]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        for (uint d = 0; d < head_dim; d++) {
            scores[simd_idx * head_dim + d] = partial_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++) {
            acc += scores[sg * head_dim + tid];
        }
        output[head * head_dim + tid] = half(acc);
    }
}

// ─── matmul_f16 (tiled GEMM, 64×32, 128 threads, 4 simdgroups) ───

kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& K_dim [[buffer(4)]],
    constant uint& N    [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Memory overlay: sa/sb share storage with tc (8KB total vs 14KB)
    // sa(4KB) + sb(2KB) during K-loop; tc(8KB) during store phase
    threadgroup float shmem[2048];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 1024);

    const uint M_base = tgpig.x * 64;
    const uint N_base = tgpig.y * 32;

    const short nr0 = short(min(uint(64), M - M_base));
    const short nr1 = short(min(uint(32), N - N_base));

    const ushort lr0 = min(ushort(tiitg >> 1), ushort(nr0 - 1));
    const ushort il0 = tiitg & 1;
    const ushort lr1 = min(ushort(tiitg >> 2), ushort(nr1 - 1));

    device const half* x = A + (uint64_t)(M_base + lr0) * K_dim + uint(il0) * 16;
    device const half* y = B + (uint64_t)(N_base + lr1) * K_dim + uint(tiitg & 3) * 8;

    const ushort a_sy = lr0 >> 3;
    const ushort a_lx = lr0 & 7;
    const ushort b_ib = ((tiitg & 3) << 2) | (lr1 >> 3);
    const ushort b_base = (b_ib << 6) | (uint(lr1 & 7) << 3);

    const ushort sg_m = sgitg & 1;
    const ushort sg_n = sgitg >> 1;

    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    for (uint k = 0; k < K_dim; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort i = 0; i < 16; i++) {
            const ushort a_sx = (il0 << 1) | (i >> 3);
            const ushort a_ib = (a_sx << 3) + a_sy;
            *(sa + (a_ib << 6) + ((i & 7) << 3) + a_lx) =
                (k + uint(il0) * 16 + i < K_dim) ? *(x + i) : half(0);
        }
        {
            threadgroup half* dst = sb + b_base;
            const uint bk = k + uint(tiitg & 3) * 8;
            if (bk + 8 <= K_dim) {
                *(threadgroup half4*)(dst)     = *(device const half4*)(y);
                *(threadgroup half4*)(dst + 4) = *(device const half4*)(y + 4);
            } else {
                for (ushort i = 0; i < 8; i++)
                    dst[i] = (bk + i < K_dim) ? *(y + i) : half(0);
            }
        }

        x += 32;
        y += 32;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + (uint(sg_m) << 8);
        threadgroup const half* lsmb = sb + (uint(sg_n) << 7);
        simdgroup_half8x8 ma[4], mb[2];

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 4; i++)
                simdgroup_load(ma[i], lsma + (i << 6), 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 2; i++)
                simdgroup_load(mb[i], lsmb + (i << 6), 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3], mc[i]);
            lsma += 512;
            lsmb += 256;
        }
    }

    // Overlay: reuse shmem as float tc[] (safe — sa/sb reads are finished)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* tc = shmem;

    for (ushort i = 0; i < 8; i++) {
        ushort m_off = (sg_m << 5) + ((i & 3) << 3);
        ushort n_off = (sg_n << 4) + ((i >> 2) << 3);
        simdgroup_store(mc[i], tc + n_off * 64 + m_off, 64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (M_base + 64 <= M && N_base + 32 <= N) {
        for (ushort idx = tiitg; idx < 512; idx += 128) {
            ushort m4 = (idx & 15) << 2;
            ushort n = idx >> 4;
            *(device half4*)(C + (uint64_t)(N_base + n) * M + M_base + m4) =
                half4(*(threadgroup const float4*)(tc + n * 64 + m4));
        }
    } else {
        for (ushort idx = tiitg; idx < 2048; idx += 128) {
            ushort m = idx & 63, n = idx >> 6;
            uint gm = M_base + m, gn = N_base + n;
            if (gm < M && gn < N)
                C[(uint64_t)gn * M + gm] = half(tc[n * 64 + m]);
        }
    }
}

// ─── matmul_q4_0 (tiled GEMM, 64×32, 128 threads, 4 simdgroups, Q4_0) ───

kernel void matmul_q4_0(
    device const uint8_t* A [[buffer(0)]],
    device const half*    B [[buffer(1)]],
    device half*          C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K_dim  [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shmem[2048];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 1024);

    const uint M_base = tgpig.x * 64;
    const uint N_base = tgpig.y * 32;

    const short nr0 = short(min(uint(64), M - M_base));
    const short nr1 = short(min(uint(32), N - N_base));

    const ushort lr0 = min(ushort(tiitg >> 1), ushort(nr0 - 1));
    const ushort half_id = tiitg & 1;
    const ushort lr1 = min(ushort(tiitg >> 2), ushort(nr1 - 1));

    device const half* y = B + (uint64_t)(N_base + lr1) * K_dim + uint(tiitg & 3) * 8;

    const uint nb = K_dim / QK4_0;
    const uint am = M_base + lr0;

    const ushort mblock = lr0 >> 3;
    const ushort m_local = lr0 & 7;
    const ushort ib_lo = (half_id << 3) + mblock;
    const ushort ib_hi = ((half_id + 2) << 3) + mblock;

    const ushort b_ib = ((tiitg & 3) << 2) | (lr1 >> 3);
    const ushort b_base = (b_ib << 6) | (uint(lr1 & 7) << 3);

    const ushort sg_m = sgitg & 1;
    const ushort sg_n = sgitg >> 1;

    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    for (uint k = 0; k < K_dim; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint blk_col = k / QK4_0;
        if (am < M) {
            device const uint8_t* blk = A + (uint64_t)am * nb * BLOCK_SIZE_Q4_0
                                           + (uint64_t)blk_col * BLOCK_SIZE_Q4_0;
            half scale = *reinterpret_cast<device const half*>(blk);
            device const uint8_t* qs = blk + 2;
            threadgroup half* dst_lo = sa + (ib_lo << 6) + m_local;
            threadgroup half* dst_hi = sa + (ib_hi << 6) + m_local;
            for (ushort j = 0; j < 8; j++) {
                uint8_t byte = qs[half_id * 8 + j];
                dst_lo[j << 3] = scale * half((int)(byte & 0x0F) - 8);
                dst_hi[j << 3] = scale * half((int)(byte >> 4) - 8);
            }
        } else {
            threadgroup half* dst_lo = sa + (ib_lo << 6) + m_local;
            threadgroup half* dst_hi = sa + (ib_hi << 6) + m_local;
            for (ushort j = 0; j < 8; j++) {
                dst_lo[j << 3] = half(0);
                dst_hi[j << 3] = half(0);
            }
        }
        {
            threadgroup half* dst = sb + b_base;
            const uint bk = k + uint(tiitg & 3) * 8;
            if (bk + 8 <= K_dim) {
                *(threadgroup half4*)(dst)     = *(device const half4*)(y);
                *(threadgroup half4*)(dst + 4) = *(device const half4*)(y + 4);
            } else {
                for (ushort i = 0; i < 8; i++)
                    dst[i] = (bk + i < K_dim) ? *(y + i) : half(0);
            }
        }

        y += 32;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + (uint(sg_m) << 8);
        threadgroup const half* lsmb = sb + (uint(sg_n) << 7);
        simdgroup_half8x8 ma[4], mb[2];

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 4; i++)
                simdgroup_load(ma[i], lsma + (i << 6), 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 2; i++)
                simdgroup_load(mb[i], lsmb + (i << 6), 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_none);
            for (ushort i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3], mc[i]);
            lsma += 512;
            lsmb += 256;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* tc = shmem;

    for (ushort i = 0; i < 8; i++) {
        ushort m_off = (sg_m << 5) + ((i & 3) << 3);
        ushort n_off = (sg_n << 4) + ((i >> 2) << 3);
        simdgroup_store(mc[i], tc + n_off * 64 + m_off, 64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (M_base + 64 <= M && N_base + 32 <= N) {
        for (ushort idx = tiitg; idx < 512; idx += 128) {
            ushort m4 = (idx & 15) << 2;
            ushort n = idx >> 4;
            *(device half4*)(C + (uint64_t)(N_base + n) * M + M_base + m4) =
                half4(*(threadgroup const float4*)(tc + n * 64 + m4));
        }
    } else {
        for (ushort idx = tiitg; idx < 2048; idx += 128) {
            ushort m = idx & 63, n = idx >> 6;
            uint gm = M_base + m, gn = N_base + n;
            if (gm < M && gn < N)
                C[(uint64_t)gn * M + gm] = half(tc[n * 64 + m]);
        }
    }
}

// ─── batch_rope ───

kernel void batch_rope(
    device half* x          [[buffer(0)]],
    constant uint& head_dim    [[buffer(1)]],
    constant uint& n_heads     [[buffer(2)]],
    constant uint& start_pos   [[buffer(3)]],
    constant float& theta_base [[buffer(4)]],
    constant uint& n_tokens    [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint pairs_per_token = n_heads * half_dim;
    uint tok_idx = tid / pairs_per_token;
    uint rem = tid % pairs_per_token;
    uint head_idx = rem / half_dim;
    uint pair_idx = rem % half_dim;

    if (tok_idx >= n_tokens) return;

    uint position = start_pos + tok_idx;
    float freq = 1.0 / pow(theta_base, float(2 * pair_idx) / float(head_dim));
    float angle = float(position) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint base_off = tok_idx * n_heads * head_dim + head_idx * head_dim;
    uint idx_re = base_off + pair_idx;
    uint idx_im = idx_re + half_dim;

    float re = float(x[idx_re]);
    float im = float(x[idx_im]);
    x[idx_re] = half(re * cos_a - im * sin_a);
    x[idx_im] = half(im * cos_a + re * sin_a);
}

// ─── bias_broadcast (add 1D bias to each row of 2D batch, in-place) ───

kernel void bias_broadcast(
    device half* batch        [[buffer(0)]],
    device const half* bias   [[buffer(1)]],
    constant uint& dim        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint col = gid % dim;
    batch[gid] = batch[gid] + bias[col];
}

// ─── Batched prefill attention (one threadgroup per (token, head), causal mask) ───

kernel void prefill_attention(
    device const half* Q_batch    [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device half*       O_batch    [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& start_position [[buffer(8)]],
    constant uint& N              [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint token_idx = tgid / n_heads;
    uint head = tgid % n_heads;
    if (token_idx >= N) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    uint q_offset = token_idx * (n_heads * head_dim) + head * head_dim;
    device const half* q_ptr = Q_batch + q_offset;

    uint causal_len = start_position + token_idx + 1;

    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float scores[8160];
    threadgroup float reduce_buf[32];

    for (uint s = tid; s < causal_len; s += tg_size) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint s = tid; s < causal_len; s += tg_size) {
        local_max = max(local_max, scores[s]);
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : -INFINITY;
        val = simd_max(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < causal_len; s += tg_size) {
        float val = exp(scores[s] - global_max);
        scores[s] = val;
        local_sum += val;
    }
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) reduce_buf[simd_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? reduce_buf[simd_lane] : 0.0f;
        val = simd_sum(val);
        reduce_buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce_buf[0];

    float partial_out[128];
    for (uint d = 0; d < head_dim; d++) partial_out[d] = 0.0f;

    for (uint s = tid; s < causal_len; s += tg_size) {
        float w = scores[s] * inv_sum;
        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            partial_out[d] += w * float(v_ptr[d]);
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        partial_out[d] = simd_sum(partial_out[d]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        for (uint d = 0; d < head_dim; d++) {
            scores[simd_idx * head_dim + d] = partial_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++) {
            acc += scores[sg * head_dim + tid];
        }
        O_batch[q_offset + tid] = half(acc);
    }
}

// ─── Flash prefill attention (online softmax, tiled KV, SIMD head_dim) ───

kernel void flash_prefill_attention(
    device const half* Q_batch    [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device half*       O_batch    [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& start_position [[buffer(8)]],
    constant uint& N              [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint token_idx = tgid / n_heads;
    uint head = tgid % n_heads;
    if (token_idx >= N) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    uint q_offset = token_idx * (n_heads * head_dim) + head * head_dim;
    device const half* q_ptr = Q_batch + q_offset;

    uint causal_len = start_position + token_idx + 1;

    uint sg_id = tid / 32;
    uint lane  = tid % 32;
    uint n_sg  = tg_size / 32;
    uint dim_base = lane * 4;

    float q_reg[4];
    for (uint i = 0; i < 4; i++)
        q_reg[i] = (dim_base + i < head_dim) ? float(q_ptr[dim_base + i]) : 0.0f;

    const uint TILE_KV = 256;
    threadgroup float tile_mem[1024];
    threadgroup float reduce_buf[32];

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tile_start = 0; tile_start < causal_len; tile_start += TILE_KV) {
        uint tile_end = min(tile_start + TILE_KV, causal_len);
        uint tile_len = tile_end - tile_start;

        for (uint s = sg_id; s < tile_len; s += n_sg) {
            uint kv_pos = tile_start + s;
            device const half* k_ptr = K_cache + kv_pos * kv_stride + kv_head * head_dim;
            float partial = 0.0f;
            for (uint i = 0; i < 4; i++)
                partial += q_reg[i] * ((dim_base + i < head_dim) ? float(k_ptr[dim_base + i]) : 0.0f);
            float dot = simd_sum(partial) * scale;
            if (lane == 0) tile_mem[s] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -INFINITY;
        for (uint s = tid; s < tile_len; s += tg_size) {
            local_max = max(local_max, tile_mem[s]);
        }
        local_max = simd_max(local_max);
        if (lane == 0) reduce_buf[sg_id] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float val = (lane < n_sg) ? reduce_buf[lane] : -INFINITY;
            val = simd_max(val);
            reduce_buf[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = reduce_buf[0];

        float new_max = max(running_max, tile_max);
        float exp_old = exp(running_max - new_max);

        float local_sum = 0.0f;
        for (uint s = tid; s < tile_len; s += tg_size) {
            float val = exp(tile_mem[s] - new_max);
            tile_mem[s] = val;
            local_sum += val;
        }
        local_sum = simd_sum(local_sum);
        if (lane == 0) reduce_buf[sg_id] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float val = (lane < n_sg) ? reduce_buf[lane] : 0.0f;
            val = simd_sum(val);
            reduce_buf[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_sum = reduce_buf[0];

        float rescale = (running_sum > 0.0f) ? exp_old : 0.0f;
        for (uint i = 0; i < 4; i++) acc[i] *= rescale;

        for (uint s = sg_id; s < tile_len; s += n_sg) {
            float w = tile_mem[s];
            uint kv_pos = tile_start + s;
            device const half* v_ptr = V_cache + kv_pos * kv_stride + kv_head * head_dim;
            for (uint i = 0; i < 4; i++)
                acc[i] += w * ((dim_base + i < head_dim) ? float(v_ptr[dim_base + i]) : 0.0f);
        }

        running_max = new_max;
        running_sum = running_sum * exp_old + tile_sum;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint i = 0; i < 4; i++) acc[i] *= inv_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; i++) {
        if (dim_base + i < head_dim)
            tile_mem[sg_id * head_dim + dim_base + i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float result = 0.0f;
        for (uint sg = 0; sg < n_sg; sg++)
            result += tile_mem[sg * head_dim + tid];
        O_batch[q_offset + tid] = half(result);
    }
}

// ─── Flash attention decode (simdgroup-parallel, online softmax) ───

kernel void flash_attention_decode(
    device const half* Q        [[buffer(0)]],
    device const half* K_cache  [[buffer(1)]],
    device const half* V_cache  [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    constant uint&     n_heads     [[buffer(4)]],
    constant uint&     n_kv_heads  [[buffer(5)]],
    constant uint&     head_dim    [[buffer(6)]],
    constant uint&     seq_len     [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head = tgid;
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    device const half* q_ptr = Q + head * head_dim;

    uint sg_id = tid / 32;
    uint lane  = tid % 32;
    uint n_sg  = tg_size / 32;
    if (n_sg > 8) return;  // 守护：threadgroup 太大则跳过
    uint dim_base = lane * 4;

    float q_reg[4];
    for (uint i = 0; i < 4; i++)
        q_reg[i] = (dim_base + i < head_dim) ? float(q_ptr[dim_base + i]) : 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint s = sg_id; s < seq_len; s += n_sg) {
        device const half* k_ptr = K_cache + s * kv_stride + kv_head * head_dim;

        float partial = 0.0f;
        for (uint i = 0; i < 4; i++)
            partial += q_reg[i] * ((dim_base + i < head_dim) ? float(k_ptr[dim_base + i]) : 0.0f);
        float dot = simd_sum(partial) * scale;

        float new_max = max(running_max, dot);
        float exp_old = exp(running_max - new_max);
        float exp_new = exp(dot - new_max);
        float rescale = (running_sum > 0.0f) ? exp_old : 0.0f;

        device const half* v_ptr = V_cache + s * kv_stride + kv_head * head_dim;
        for (uint i = 0; i < 4; i++) {
            float v_val = (dim_base + i < head_dim) ? float(v_ptr[dim_base + i]) : 0.0f;
            acc[i] = acc[i] * rescale + exp_new * v_val;
        }

        running_max = new_max;
        running_sum = running_sum * exp_old + exp_new;
    }

    threadgroup float sg_max[8];
    threadgroup float sg_sum[8];
    threadgroup float sg_acc[8 * 128];
    threadgroup float tg_global_max;
    threadgroup float tg_inv_sum;

    if (lane == 0) {
        sg_max[sg_id] = running_max;
        sg_sum[sg_id] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gmax = -INFINITY;
        for (uint i = 0; i < n_sg; i++)
            gmax = max(gmax, sg_max[i]);
        tg_global_max = gmax;
        float gsum = 0.0f;
        for (uint i = 0; i < n_sg; i++) {
            if (sg_sum[i] > 0.0f)
                gsum += sg_sum[i] * exp(sg_max[i] - gmax);
        }
        tg_inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_global_max;
    float inv_sum = tg_inv_sum;
    float my_rescale = (running_sum > 0.0f) ? exp(running_max - global_max) : 0.0f;

    for (uint i = 0; i < 4; i++)
        sg_acc[sg_id * 128 + dim_base + i] = acc[i] * my_rescale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float result = 0.0f;
        for (uint i = 0; i < n_sg; i++)
            result += sg_acc[i * 128 + tid];
        output[head * head_dim + tid] = half(result * inv_sum);
    }
}

// ─── Fused SiLU + elementwise multiply ───

kernel void silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half* output     [[buffer(2)]],
    constant uint& N        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = float(gate[tid]);
    float s = x / (1.0f + exp(-x));
    output[tid] = half(s * float(up[tid]));
}

// ─── scatter_kv (GPU-side KV buffer copy) ───

kernel void scatter_kv(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& offset  [[buffer(2)]],
    constant uint& count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        dst[offset + tid] = src[tid];
    }
}

// ─── argmax_f16 (single-threadgroup reduce for greedy decode) ───

kernel void argmax_f16(
    device const half* input [[buffer(0)]],
    device uint* result      [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = tid; i < N; i += tg_size) {
        float v = float(input[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    local_max = simd_max(local_max);
    uint simd_idx  = tid / 32;
    uint simd_lane = tid % 32;

    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(local_max, offset);
        uint other_idx  = simd_shuffle_down(local_idx, offset);
        if (other_val > local_max) { local_max = other_val; local_idx = other_idx; }
    }

    threadgroup float shared_val[32];
    threadgroup uint  shared_idx[32];
    uint n_sg = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared_val[simd_idx] = local_max;
        shared_idx[simd_idx] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        float val = (simd_lane < n_sg) ? shared_val[simd_lane] : -INFINITY;
        uint  idx = (simd_lane < n_sg) ? shared_idx[simd_lane] : 0;

        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(val, offset);
            uint  other_idx = simd_shuffle_down(idx, offset);
            if (other_val > val) { val = other_val; idx = other_idx; }
        }

        if (simd_lane == 0) {
            result[0] = idx;
        }
    }
}

// ─── matmul_f16_naive (1 thread per output element, for small matrices) ───

kernel void matmul_f16_naive(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& K_dim [[buffer(4)]],
    constant uint& N    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= M || row >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K_dim; k += 4) {
        float4 a = float4(
            A[uint64_t(col) * K_dim + k],
            A[uint64_t(col) * K_dim + k + 1],
            (k + 2 < K_dim) ? float(A[uint64_t(col) * K_dim + k + 2]) : 0.f,
            (k + 3 < K_dim) ? float(A[uint64_t(col) * K_dim + k + 3]) : 0.f
        );
        float4 b = float4(
            B[uint64_t(row) * K_dim + k],
            B[uint64_t(row) * K_dim + k + 1],
            (k + 2 < K_dim) ? float(B[uint64_t(row) * K_dim + k + 2]) : 0.f,
            (k + 3 < K_dim) ? float(B[uint64_t(row) * K_dim + k + 3]) : 0.f
        );
        sum += dot(a, b);
    }
    C[uint64_t(row) * M + col] = half(sum);
}

// ─── matmul_q4_0_naive (1 thread per output element, for small matrices) ───

kernel void matmul_q4_0_naive(
    device const uint8_t* A [[buffer(0)]],
    device const half*    B [[buffer(1)]],
    device half*          C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K_dim  [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= M || row >= N) return;

    const uint nb = K_dim / QK4_0;
    float sum = 0.0f;

    for (uint blk = 0; blk < nb; blk++) {
        device const uint8_t* block = A + uint64_t(col) * nb * BLOCK_SIZE_Q4_0
                                        + uint64_t(blk) * BLOCK_SIZE_Q4_0;
        half scale = *reinterpret_cast<device const half*>(block);
        device const uint8_t* qs = block + 2;
        device const half* Brow = B + uint64_t(row) * K_dim + blk * QK4_0;

        float blk_sum = 0.0f;
        for (ushort j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            int lo = int(byte & 0x0F) - 8;
            int hi = int(byte >> 4) - 8;
            blk_sum += float(lo) * float(Brow[j]) + float(hi) * float(Brow[j + 16]);
        }
        sum += float(scale) * blk_sum;
    }
    C[uint64_t(row) * M + col] = half(sum);
}
)";

}  // namespace mugen::metal
