# Model Support

> Supported architectures, quantization formats, memory requirements, and model acquisition.

## Supported Architectures

### Dense Models

| Architecture | Examples | Status | Notes |
|-------------|----------|--------|-------|
| **Llama** | Llama 3.2 (1B, 3B), Llama 3.1 (8B), Llama 2 (7B, 13B) | Stable | GQA attention, SiLU-gated FFN, ChatML + Llama 3 templates |
| **Qwen** | Qwen 2.5 (0.5B, 3B, 7B, 14B) | Stable | GQA attention, SiLU-gated FFN, ChatML template |

### MoE Models

| Architecture | Examples | Status | Notes |
|-------------|----------|--------|-------|
| **OLMoE** | OLMoE-1B-7B (64 experts, top-8) | Stable | Standard softmax routing |
| **DeepSeek V2** | DeepSeek-V2-Lite (15.7B) | Beta | MLA absorbed attention, shared + routed experts, YaRN RoPE |
| **DeepSeek V3** | DeepSeek-V3 (671B, 256 experts, top-8) | In Progress | Grouped sigmoid routing, router bias, MLA, SSD offload required |

### Architecture Feature Matrix

| Feature | Llama | Qwen | OLMoE | DeepSeek V2 | DeepSeek V3 |
|---------|-------|------|-------|-------------|-------------|
| Attention | GQA | GQA | MHA | MLA | MLA |
| FFN | Dense | Dense | MoE | MoE + Shared | MoE + Shared |
| Router | — | — | Softmax | Softmax | Grouped Sigmoid |
| RoPE | Standard | Standard | Standard | YaRN | YaRN |
| KV compression | 1× | 1× | 1× | 42× (MLA) | 42× (MLA) |

---

## GGUF Quantization Format Compatibility

| Format | Bits/Weight | Description | Decode (matvec) | Prefill (matmul) | Dequant |
|--------|-----------|-------------|-----------------|-----------------|---------|
| **F16** | 16 | Half-precision float | `matvec_f16` | `matmul_f16` | N/A |
| **Q4_0** | 4.5 | 4-bit uniform, scale only | `matvec_q4_0` | `matmul_q4_0` | `dequantize_q4_0` |
| **Q8_0** | 8.5 | 8-bit uniform, scale only | `matvec_q8_0` | — | `dequantize_q8_0` |
| **Q4_K** | 4.5 | 4-bit k-quant, scale + min | `matvec_q4_k` | — | `dequantize_q4_k` |

> Q4_0 is the recommended default for best performance/quality tradeoff. Q4_K provides slightly better quality at the same speed. Q8_0 offers higher fidelity at ~2× memory cost. F16 is primarily used for small models or debugging.

### Format Support by Model Size

| Model Size | Recommended Format | Reason |
|-----------|-------------------|--------|
| ≤ 3B | Q8_0 or F16 | Fits easily in memory, maximize quality |
| 7B–14B | Q4_0 or Q4_K | Good quality/memory balance |
| 30B+ | Q4_0 | Memory constrained |
| 300B+ (MoE) | Q4_0 with SSD offload | Requires USPP pipeline |

---

## Memory Requirements

### Dense Models

| Model | Parameters | Q4_0 Size | Q8_0 Size | F16 Size | Min RAM |
|-------|-----------|-----------|-----------|----------|---------|
| Llama 3.2 1B | 1.2B | ~0.7 GB | ~1.3 GB | ~2.4 GB | 16 GB |
| Qwen 2.5 3B | 3.1B | ~1.8 GB | ~3.3 GB | ~6.2 GB | 16 GB |
| Llama 3.1 8B | 8.0B | ~4.5 GB | ~8.5 GB | ~16 GB | 32 GB |
| Qwen 2.5 14B | 14.2B | ~8.0 GB | ~15 GB | ~28 GB | 32 GB |

### MoE Models

| Model | Total Params | Active Params | Q4_0 Size | Q8_0 Size | Min RAM | Notes |
|-------|-------------|---------------|-----------|-----------|---------|-------|
| OLMoE-1B-7B | 6.9B | 1.3B | ~4.0 GB | ~7.5 GB | 32 GB | 64 experts, top-8 |
| DeepSeek-V2-Lite | 15.7B | 2.4B | ~9.0 GB | ~17 GB | 64 GB | MLA + shared expert |
| DeepSeek-V3 | 671B | ~37B | ~340 GB | ~670 GB | 64 GB* | *SSD offload required |

> *DeepSeek V3 at Q4_0 requires ~340 GB storage. With USPP and SSD offload, only ~35 GB of expert weights need to be resident in memory at any time (active experts + pinned hot experts + KV cache).

### Memory Budget Breakdown (64 GB System, DeepSeek V3 Q4)

```
┌──────────────────────────────────┐
│ Total Unified Memory:    64 GB   │
├──────────────────────────────────┤
│ macOS + System:          ~8 GB   │
│ KV Cache (MLA compressed): ~2 GB │
│ Embeddings + Norms:      ~1 GB   │
│ Pinned Hot Experts:      ~5 GB   │
│ Active Expert Buffer:   ~15 GB   │
│ Staging Expert Buffer:  ~15 GB   │
│ GPU Scratch:             ~2 GB   │
│ Headroom:               ~16 GB   │
└──────────────────────────────────┘
```

---

## Performance Baselines

Benchmarked on **Mac mini M4 Pro** (14-core CPU, 20-core GPU, 64 GB unified memory).

### Dense Models

| Model | Quant | TTFT | Prefill | Decode | Notes |
|-------|-------|------|---------|--------|-------|
| Llama 3.2 1B | Q4_0 | 69 ms | 694 tok/s | 266.2 tok/s | Mega-chain decode |
| Qwen 2.5 7B | Q4_0 | 163 ms | 295 tok/s | 56.1 tok/s | Mega-chain decode |

### MoE Models

| Model | Quant | TTFT | Prefill | Decode | Notes |
|-------|-------|------|---------|--------|-------|
| OLMoE-1B-7B | Q4_0 | — | — | 130.7 tok/s | Batch routing (10.4× over naive) |
| DeepSeek-V2-Lite | Q8_0 | 9.1 s | 5.4 tok/s | 4.9 tok/s | CPU MLA (GPU kernel WIP) |

> DeepSeek V2-Lite decode performance will reach 30+ tok/s once the GPU MLA attention kernel is complete. Current bottleneck is CPU-side absorbed attention.

### North Star Target

| Model | Quant | Target Decode | Status |
|-------|-------|--------------|--------|
| DeepSeek V3 (671B) | Q4 | ≥ 8 tok/s | In Progress |

---

## Model Acquisition

### From Hugging Face

Most GGUF models are available on Hugging Face. Use any HTTPS download tool:

```sh
# Create model directory
mkdir -p ~/.mugen/models

# Download a model (example: Qwen 2.5 7B Q4_0)
curl -L -o ~/.mugen/models/qwen2.5-7b-q4_0.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf"

# For large models, use huggingface-cli for resumable downloads
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_0.gguf \
  --local-dir ~/.mugen/models
```

### Sharded (Split) Models

Large models may be split across multiple GGUF files:

```
deepseek-v3-q4_0-00001-of-00050.gguf
deepseek-v3-q4_0-00002-of-00050.gguf
...
deepseek-v3-q4_0-00050-of-00050.gguf
```

Mugen automatically detects split files. Point to any shard and all others in the same directory will be found:

```sh
./build/src/mugen-cli chat ~/.mugen/models/deepseek-v3-q4_0-00001-of-00050.gguf
```

### Recommended Models for Getting Started

| Use Case | Model | Size | Download |
|----------|-------|------|----------|
| Quick test | Llama 3.2 1B Instruct Q4_0 | ~0.7 GB | [HuggingFace](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| General use | Qwen 2.5 7B Instruct Q4_0 | ~4.5 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| MoE testing | OLMoE-1B-7B Q4_0 | ~4.0 GB | [HuggingFace](https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF) |
| MLA testing | DeepSeek-V2-Lite Q8_0 | ~17 GB | [HuggingFace](https://huggingface.co/TheBloke/deepseek-llm-7b-chat-GGUF) |

### Model Directory

By default, Mugen looks for models in `~/.mugen/models/`. Override with:

```sh
export MUGEN_MODEL_DIR=/path/to/models
```

### Verifying a Model

```sh
# Check model metadata
./build/src/mugen-cli info <model.gguf>

# List all downloaded models
./build/src/mugen-cli list
```

---

## Converting Models to GGUF

Mugen reads standard GGUF files. To convert from Hugging Face format, use llama.cpp's conversion tools:

```sh
# Clone llama.cpp (for conversion only)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HF model to GGUF
python convert_hf_to_gguf.py /path/to/hf-model --outfile model-f16.gguf

# Quantize to Q4_0
./build/bin/llama-quantize model-f16.gguf model-q4_0.gguf Q4_0
```

> Mugen is fully compatible with GGUF files produced by llama.cpp. No additional conversion is needed.

---

## Planned Model Support

| Architecture | Timeline | Notes |
|-------------|----------|-------|
| DeepSeek V3 (full 671B) | Phase 3 | GPU MLA kernel + SSD offload |
| Mixtral | Future | Standard MoE, similar to OLMoE path |
| Qwen 3 MoE | Future | Pending GGUF availability |
| Gemma 2 | Future | Sliding window attention variant |
