# MoE Inference

Run Mixture-of-Experts models on Apple Silicon with Mugen's Unified Shared-memory Parallel Pipeline (USPP).

## Background

A Mixture-of-Experts (MoE) layer replaces the single feed-forward network in a Transformer block with *N* expert networks and a lightweight router. For each token, the router selects the top-*k* experts (typically 2 out of 8–64), so only a fraction of the parameters are active per forward pass. This gives MoE models a favorable compute-to-parameter ratio: a 47B-parameter model with 8 active experts may run at speeds closer to a 12B dense model.

## How Mugen Handles MoE

1. **Expert routing on Metal.** The gating network and top-k selection run as fused Metal kernels, avoiding CPU–GPU round-trips.
2. **USPP (Unified Shared-memory Parallel Pipeline).** Mugen pipelines expert execution across Metal command buffers. While one group of experts computes on the GPU, the next group's weights are prefetched from unified memory — hiding memory latency behind compute.
3. **Sparse KV cache.** Only active experts contribute to the KV cache, reducing memory pressure proportionally.

## 1. Supported Models

| Model | Parameters | Active | Experts | Min Memory |
|-------|-----------|--------|---------|------------|
| [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF) | 7B | 1B | 64×8 | ~5 GB (Q4) |
| [DeepSeek-V2-Lite](https://huggingface.co/TheBloke/DeepSeek-V2-Lite-GGUF) | 16B | ~2.4B | 64×6 | ~10 GB (Q4) |
| [Mixtral-8x7B](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF) | 47B | ~12B | 8×2 | ~27 GB (Q4) |

## 2. Download and Run OLMoE

```console
$ huggingface-cli download allenai/OLMoE-1B-7B-0924-GGUF \
    olmoe-1b-7b-0924.Q4_K_M.gguf \
    --local-dir ./models

$ mugen-cli chat --model ./models/olmoe-1b-7b-0924.Q4_K_M.gguf
```

```console
[mugen] Loading olmoe-1b-7b-0924.Q4_K_M.gguf … done (0.9 s)
[mugen] Architecture: MoE (64 experts, top-8 routing)
[mugen] Active parameters: ~1B per token
[mugen] USPP pipeline: enabled (4 stages)

You> What makes MoE models efficient?

Assistant> MoE models route each token to a small subset of specialized
expert networks rather than passing it through every parameter. This
sparse activation means the model can store far more knowledge in its
total parameters while keeping per-token compute — and therefore
latency — comparable to a much smaller dense model.

[mugen] 52 tokens | 87.3 tok/s | experts hit: 8/64
```

## 3. Memory Estimation

Estimate unified memory requirements before downloading:

```
Memory ≈ model_size_on_disk + (ctx_size × active_params × kv_bytes)
```

Practical estimates for Q4_K_M quantization with a 4096-token context:

| Model | Weights | KV Cache | Total |
|-------|---------|----------|-------|
| OLMoE-1B-7B | 4.2 GB | 0.5 GB | ~5 GB |
| DeepSeek-V2-Lite | 8.8 GB | 1.1 GB | ~10 GB |
| Mixtral-8x7B | 24.6 GB | 2.8 GB | ~28 GB |

> Mixtral-8x7B at Q4 fits on 32 GB machines but leaves limited headroom. Use Q3_K_S (~20 GB) if memory is tight.

## 4. Performance: MoE vs. Dense

Comparison on Apple M2 Max (32 GB), prompt 512 tokens, generate 256 tokens:

| Model | Type | Params (active) | Decode (tok/s) | Quality (MMLU) |
|-------|------|-----------------|----------------|----------------|
| Llama-2-7B | Dense | 7B | 62.1 | 45.3 |
| OLMoE-1B-7B | MoE | 1B of 7B | 87.3 | 52.1 |
| Llama-2-13B | Dense | 13B | 31.4 | 54.8 |
| DeepSeek-V2-Lite | MoE | 2.4B of 16B | 58.6 | 55.9 |

**Takeaway:** MoE models achieve higher quality at faster decode speeds than dense models of comparable accuracy, because only a fraction of parameters are active per token.

## 5. Expert Routing Diagnostics

Use `--verbose` to inspect per-layer expert utilization:

```console
$ mugen-cli chat --model ./models/olmoe-1b-7b-0924.Q4_K_M.gguf --verbose
```

```console
[mugen] Layer  0: experts [3,7,12,19,24,31,48,55] load 12.5%
[mugen] Layer  1: experts [1,5,14,22,29,33,41,60] load 12.5%
  …
[mugen] Expert utilization: mean 12.5%, std 1.8% (well-balanced)
```

Unbalanced routing (high standard deviation) may indicate a poorly trained gating network or degenerate expert collapse. Mugen reports this as a warning.

## 6. Tips

- **USPP tuning.** Override the pipeline depth with `--uspp-stages N`. The default (auto) picks a value based on expert count and available GPU cores.
- **Mixed quantization.** Mugen supports per-expert quantization in GGUF. Frequently activated experts can use higher precision (Q5/Q6) while rare experts use Q3/Q4 — a good quality/memory trade-off.
- **Batch inference.** When serving via the API server, MoE models benefit from request batching: tokens from different requests may activate different experts, keeping GPU utilization high.
