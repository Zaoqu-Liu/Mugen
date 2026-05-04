#include <metal_stdlib>
using namespace metal;

// Quantized matrix-vector multiply kernel stub.
// Will be replaced by fused dequant + GEMV kernels for Q4_K / Q8_0 formats,
// tiled for the Apple GPU SIMD width (32 threads per SIMD group).

kernel void matvec_f16(
    device const half*   A [[buffer(0)]],
    device const half*   x [[buffer(1)]],
    device       half*   y [[buffer(2)]],
    constant     uint&   M [[buffer(3)]],
    constant     uint&   K [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        acc += float(A[tid * K + j]) * float(x[j]);
    }
    y[tid] = half(acc);
}
