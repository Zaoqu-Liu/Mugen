# Performance Guide

This guide covers how to get the most out of Mugen on Apple Silicon. It assumes you have a working build — see [BUILDING.md](BUILDING.md) if not.

## Memory Hierarchy

Mugen operates across three memory tiers. Understanding them is key to performance tuning.

```
┌─────────────────────────────────────────────┐
│  Tier 0: GPU-Resident (Metal Buffers)       │  ← Fastest (~2 TB/s bandwidth)
│  Shared attention weights, active experts    │  ← Always in unified memory
├─────────────────────────────────────────────┤
│  Tier 1: Memory-Mapped Buffer Pool          │  ← Fast (~100 GB/s)
│  Recently used experts under LRU eviction   │  ← mmap'd from GGUF, promoted on demand
├─────────────────────────────────────────────┤
│  Tier 2: NVMe SSD (Cold Storage)            │  ← ~7 GB/s sequential read
│  Full model file, experts fetched via USPP  │  ← Latency hidden by speculative prefetch
└─────────────────────────────────────────────┘
```

**Tier 0** holds weights that are used every token — attention projections, layer norms, and the top-K active experts for the current layer. These live in wired `MTLBuffer` objects and are never evicted.

**Tier 1** is a pool of memory-mapped expert weight pages. When the router selects an expert, Mugen checks if its weights are already mapped. If so, the page is promoted to a Metal buffer (zero-copy on unified memory). A heat-scored LRU policy evicts cold experts when the pool exceeds budget.

**Tier 2** is the GGUF file on NVMe SSD. The USPP pipeline reads expert weights two layers ahead of need, so by the time the GPU reaches a layer, the required experts are already in Tier 1.

### Memory Budget Rule of Thumb

For a model of size `S` at quantization `Q`:

| What | Estimate |
|------|----------|
| Shared weights (attention, norms, embeddings) | ~15–25% of total model size |
| Active expert buffer pool | Configurable; default = available RAM − 8 GB |
| KV cache | Depends on context length; ~1–4 GB typical |
| System overhead | Reserve 4–8 GB for macOS and other apps |

**Example:** DeepSeek V2-Lite Q8_0 (~30 GB on disk) on a 64 GB Mac:
- Shared weights: ~6 GB (Tier 0)
- Expert pool budget: ~45 GB (Tier 1)
- KV cache: ~2 GB
- System: ~8 GB free
- Remaining experts: streamed from SSD (Tier 2) via USPP

## Quantization Format Selection

| Format | Bits/Weight | Quality | Speed | Best For |
|--------|:-----------:|---------|-------|----------|
| F16 | 16 | Reference | Slowest | Accuracy validation, small models |
| Q8_0 | 8 | Near-lossless | Medium | Best quality-speed tradeoff for models that fit in RAM |
| Q4_K | 4.5 | Good | Fast | Recommended default for MoE models |
| Q4_0 | 4 | Acceptable | Fastest | Maximum model size in limited RAM |

**Guidance:**
- Start with **Q4_K** — it offers the best balance for MoE workloads.
- Use **Q8_0** only if the model fits comfortably in RAM (shared weights + expert pool < available RAM − 8 GB).
- Use **Q4_0** when you need to squeeze the largest possible model onto your hardware.
- **F16** is mainly for correctness testing; it is 4× the size of Q4_0 and significantly slower.

## Speculative Decoding Tuning

The USPP pipeline has several tunable parameters that affect the throughput–latency tradeoff.

### Draft Model Selection

The draft model generates candidate tokens and, critically, predicts expert routing for the target model. A good draft model should:
1. Be small enough to run at near-zero cost (0.5B–3B parameters)
2. Share the same tokenizer as the target model
3. Produce reasonably accurate next-token predictions (acceptance rate > 40%)

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--draft-tokens` | 4 | Number of candidate tokens per speculation round |
| `--prefetch-depth` | 2 | How many layers ahead to prefetch experts |
| `--expert-budget` | auto | Max memory for Tier 1 expert pool (bytes) |
| `--batch-size` | 1 | Decode batch size (higher = better throughput, more memory) |

**Tips:**
- Increase `--draft-tokens` if the draft model has high acceptance rate (> 60%). Diminishing returns above 8.
- `--prefetch-depth 2` is optimal for most NVMe SSDs (~100 μs latency). Increase to 3 if using slower external storage.
- Set `--expert-budget` explicitly if running alongside other GPU workloads.

## System-Level Optimization

### Disable macOS Swap

macOS swap can interfere with Mugen's memory management. When swap is active, the OS may page out expert buffers that Mugen expects to be resident, causing unpredictable latency spikes.

```sh
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
```

To re-enable after your session:
```sh
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
```

> **Warning:** Disabling swap means your system will hard-fail if it runs out of physical memory. Close unnecessary applications first.

### Ensure SSD TRIM

TRIM keeps SSD performance consistent over time by allowing the drive to reclaim deleted blocks.

```sh
# Check TRIM status
system_profiler SPNVMeDataType | grep TRIM

# Enable if not already on
sudo trimforce enable
```

### Close Background Applications

Unified memory is shared between CPU and GPU. Every byte used by other apps is a byte unavailable for Mugen's expert pool. The biggest offenders:
- Web browsers (Chrome, Safari with many tabs)
- Docker Desktop
- Electron apps (Slack, Discord, VS Code)

### Thermal Management

Sustained inference generates heat. If the SoC throttles, throughput drops. For long benchmarks:
- Ensure adequate ventilation (Mac mini: do not stack objects on top)
- Monitor thermal state with `mugen-cli doctor` or `sudo powermetrics --samplers smc`
- Consider an external fan for Mac mini during extended runs

## Profiling

### Built-in Benchmarks

```sh
# Quick benchmark (prefill + decode, reports tok/s and TTFT)
./build/src/mugen-cli bench <model.gguf>

# Detailed benchmark with specific parameters
./build/src/mugen-cli bench <model.gguf> \
    --prompt-tokens 512 \
    --gen-tokens 128 \
    --repeat 5
```

### Metal GPU Profiling

Use Xcode's Metal System Trace to analyze kernel execution:

1. Open **Instruments.app** (ships with Xcode)
2. Select the **Metal System Trace** template
3. Attach to the `mugen-cli` process
4. Look for:
   - Kernel execution gaps (indicates CPU-side bottleneck or SSD stall)
   - Long `waitUntilCompleted` blocks (GPU is idle, waiting for data)
   - Buffer allocation spikes (expert pool thrashing)

### Memory Monitoring

```sh
# System-wide memory pressure
memory_pressure

# Mugen-specific diagnostics
./build/src/mugen-cli doctor

# Detailed per-process memory (look for "Mugen" or "mugen-cli")
sudo footprint -a -p $(pgrep mugen-cli)
```

### Micro-benchmarks

Individual kernel performance can be tested with the micro-benchmark suite:

```sh
# Build benchmarks
cmake -B build -DMUGEN_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# Run all micro-benchmarks
./build/bench/micro/mugen-bench-all

# Run a specific kernel benchmark
./build/bench/micro/mugen-bench-gemv-q4k
```

## Quick Checklist

Before reporting performance issues, verify:

- [ ] Built with `-DCMAKE_BUILD_TYPE=Release` (not Debug)
- [ ] macOS swap is disabled
- [ ] SSD TRIM is enabled
- [ ] No other GPU-intensive apps are running
- [ ] Sufficient free unified memory (`mugen-cli doctor`)
- [ ] Using appropriate quantization for your RAM budget
- [ ] Draft model loaded for speculative decoding (MoE models)
