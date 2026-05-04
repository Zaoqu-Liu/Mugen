# Speculative Decoding

Accelerate text generation by using a small *draft* model to propose tokens that a larger *target* model verifies in parallel. On Apple Silicon, this can yield 1.5–2× throughput with no quality loss.

## How It Works

1. The **draft model** (small, fast) generates *K* candidate tokens autoregressively.
2. The **target model** (large, accurate) scores all *K* tokens in a single forward pass.
3. Tokens accepted by the target model are kept; the first rejected token triggers a resample from the target distribution.
4. Because the target model processes a batch rather than single tokens, Metal GPU utilization improves significantly.

The output distribution is **mathematically identical** to running the target model alone — speculative decoding is a pure speed optimization.

## 1. Download Models

You need a draft/target pair from the same model family:

```console
# Draft model — small and fast
$ huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir ./models

# Target model — large and accurate
$ huggingface-cli download TheBloke/Llama-2-13B-Chat-GGUF \
    llama-2-13b-chat.Q4_K_M.gguf \
    --local-dir ./models
```

## 2. Run with Speculative Decoding

```console
$ mugen-cli chat \
    --model ./models/llama-2-13b-chat.Q4_K_M.gguf \
    --draft-model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --draft-tokens 8
```

- `--draft-tokens` sets how many candidate tokens the draft model proposes per step (default: 5). Higher values improve throughput if the draft model aligns well with the target, but waste compute when acceptance rates are low.

```console
[mugen] Loading target: llama-2-13b-chat.Q4_K_M.gguf … done (2.4 s)
[mugen] Loading draft:  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf … done (0.3 s)
[mugen] Speculative decoding: K=8, target 13B, draft 1.1B

You> Write a short poem about silicon.

Assistant> Beneath the orchard canopy it hides,
a lattice hewn from ancient sandy tides.
No voice, no breath — yet thoughts cascade and flow
through pathways only electron currents know.

[mugen] 38 tokens | 54.7 tok/s | acceptance rate: 78.2%
```

## 3. Performance Comparison

Benchmark results on an M2 Max (32 GB) generating 256 tokens with a 512-token prompt:

| Configuration | Decode (tok/s) | TTFT (ms) | Memory |
|---------------|---------------|-----------|--------|
| 13B target only | 31.4 | 189 | 7.8 GB |
| 13B + 1.1B draft (K=5) | 48.1 | 204 | 8.4 GB |
| 13B + 1.1B draft (K=8) | 54.7 | 210 | 8.4 GB |

**Observations:**

- Decode throughput improved **1.7×** with K=8.
- TTFT increases slightly because the draft model must also be loaded, but this is a one-time cost.
- Memory overhead is modest — the draft model adds ~0.6 GB.

## 4. Choosing Draft Tokens (K)

| K | Best for | Trade-off |
|---|----------|-----------|
| 3–5 | Mismatched pairs, code generation | Conservative — fewer wasted proposals |
| 6–8 | Same-family pairs, natural language | Good balance for most workloads |
| 10+ | Very high acceptance-rate pairs | Diminishing returns; profile first |

Monitor the **acceptance rate** in verbose output (`--verbose`). If it drops below 60%, reduce K.

## 5. Tips

- **Same tokenizer required.** Draft and target models must share the same tokenizer. Mugen validates this at startup.
- **Memory planning.** Both models reside in unified memory simultaneously. Ensure you have headroom beyond their combined size for KV caches.
- **Batch benchmarking.** Use `mugen-cli bench --draft-model` to run controlled comparisons (see the [benchmarking](../benchmarking/) example).
