# Benchmarking

Measure inference performance with `mugen-cli bench` and learn how to interpret the results.

## Key Metrics

| Metric | Definition |
|--------|-----------|
| **TTFT** (Time to First Token) | Wall-clock time from request submission to the first generated token. Reflects prompt processing (prefill) latency. |
| **Prefill** (tok/s) | Throughput of the prompt encoding phase. Bound by memory bandwidth on Apple Silicon. |
| **Decode** (tok/s) | Throughput of the autoregressive generation phase. The number users perceive as "speed." |
| **Peak Memory** | High-water mark of unified memory consumption during the run. |

## 1. Quick Benchmark

```console
$ mugen-cli bench --model ./models/llama-2-7b-chat.Q4_K_M.gguf
```

```console
[mugen] Benchmark: llama-2-7b-chat.Q4_K_M.gguf
[mugen] Metal backend: Apple M2 Max, 32 GB unified memory
[mugen] Warmup: 1 iteration

  Prompt length │ Generated │   TTFT │ Prefill │ Decode │ Peak Mem
 ───────────────┼───────────┼────────┼─────────┼────────┼─────────
        128 tok │   256 tok │  38 ms │ 3368 t/s│ 62.1 t/s│  5.1 GB
        512 tok │   256 tok │ 141 ms │ 3631 t/s│ 61.8 t/s│  5.3 GB
       2048 tok │   256 tok │ 548 ms │ 3735 t/s│ 60.4 t/s│  5.9 GB

[mugen] Done. 3 configurations, best-of-3 each.
```

## 2. Custom Configurations

Fine-tune what gets measured:

```console
$ mugen-cli bench \
    --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt-tokens 128,512,1024,2048,4096 \
    --gen-tokens 128,512 \
    --runs 5 \
    --warmup 2 \
    --output results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt-tokens` | `128,512,2048` | Comma-separated prompt lengths to test |
| `--gen-tokens` | `256` | Number of tokens to generate per run |
| `--runs` | `3` | Iterations per configuration (reports median) |
| `--warmup` | `1` | Warmup iterations (excluded from results) |
| `--output` | *(stdout)* | Write JSON results to a file |
| `--draft-model` | *(none)* | Benchmark with speculative decoding enabled |

## 3. Speculative Decoding Benchmark

```console
$ mugen-cli bench \
    --model ./models/llama-2-13b-chat.Q4_K_M.gguf \
    --draft-model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --draft-tokens 5,8 \
    --prompt-tokens 512 \
    --gen-tokens 256
```

```console
  Draft K │ Decode │ Accept Rate │ Speedup
 ─────────┼────────┼─────────────┼────────
     none │ 31.4 t/s│         — │    1.0×
        5 │ 48.1 t/s│      72.4% │    1.53×
        8 │ 54.7 t/s│      78.2% │    1.74×
```

## 4. Comparing with llama.cpp

Run equivalent benchmarks in both engines to get an apples-to-apples comparison.

### llama.cpp

```console
$ ./llama-bench \
    -m ./models/llama-2-7b-chat.Q4_K_M.gguf \
    -p 512 -n 256 -r 3
```

### Mugen

```console
$ mugen-cli bench \
    --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt-tokens 512 \
    --gen-tokens 256 \
    --runs 3
```

### Side-by-side (M2 Max, 32 GB)

| Engine | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) |
|--------|----------------|----------------|-----------|
| llama.cpp (Metal) | 3280 | 55.2 | 156 |
| Mugen | 3631 | 61.8 | 141 |
| **Delta** | **+10.7%** | **+12.0%** | **−9.6%** |

> Performance varies by hardware, OS version, and quantization. Always benchmark on your own machine for authoritative numbers.

## 5. Reading the JSON Output

```console
$ mugen-cli bench --model ./models/llama-2-7b-chat.Q4_K_M.gguf --output results.json
$ python3 -m json.tool results.json
```

```json
{
  "model": "llama-2-7b-chat.Q4_K_M.gguf",
  "backend": "metal",
  "device": "Apple M2 Max",
  "memory_gb": 32,
  "runs": [
    {
      "prompt_tokens": 512,
      "gen_tokens": 256,
      "ttft_ms": 141.3,
      "prefill_tok_s": 3631.2,
      "decode_tok_s": 61.8,
      "peak_memory_gb": 5.3
    }
  ]
}
```

## 6. Tips

- **Close background apps.** Memory-bandwidth-sensitive workloads benefit from minimal contention on unified memory.
- **Thermal throttling.** Sustained benchmarks on laptops may show declining throughput. Use `--runs 5` and report the median.
- **Quantization sweep.** Benchmark Q3_K_S, Q4_K_M, Q5_K_M, and Q6_K back-to-back to find your quality/speed sweet spot.
