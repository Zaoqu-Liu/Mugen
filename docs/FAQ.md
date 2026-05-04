# Frequently Asked Questions

## System Requirements

### What hardware do I need?

Any Apple Silicon Mac running macOS 14 (Sonoma) or later. Minimum 32 GB unified memory for small models; 64 GB recommended for MoE models like DeepSeek V2-Lite. For the full DeepSeek V3 (671B), 64 GB RAM + fast NVMe SSD is required — the SSD serves as extended VRAM via USPP.

### Does Mugen work on Intel Macs?

No. Mugen requires Apple Silicon (M1 or later) for its Metal GPU kernels and unified memory architecture. There is no CPU-only fallback.

### What macOS version is required?

macOS 14 (Sonoma) or later. Metal 3 features used by some kernels require macOS 14+.

### Do I need Xcode installed?

Yes. The Metal shader compiler (`metal` and `metallib`) ships with Xcode. The full Xcode IDE is not needed — Xcode Command Line Tools are sufficient for building, but the Metal toolchain requires the Xcode app bundle to be present.

---

## How is Mugen different?

### How is Mugen different from llama.cpp?

llama.cpp is a mature, cross-platform inference engine with broad hardware support. If your model fits in RAM, llama.cpp is an excellent choice.

Mugen focuses on a specific niche: running MoE models that **exceed available RAM** by treating NVMe SSD as transparent VRAM. Its USPP pipeline (speculative decoding + route prediction + NVMe prefetch) hides SSD latency behind GPU compute, enabling 671B+ parameter models on a 64 GB Mac.

| Concern | llama.cpp | Mugen |
|---------|-----------|-------|
| Model fits in RAM | Best choice — broader quant formats, cross-platform | Works, but no advantage |
| Model exceeds RAM | Not supported | Core use case — SSD offload via USPP |

### How is Mugen different from Ollama?

Ollama wraps llama.cpp in a user-friendly package with model management. It shares llama.cpp's RAM limitation. Mugen is a lower-level engine focused on pushing the boundary of what hardware can run which models.

### How is Mugen different from MLX?

MLX is Apple's general-purpose ML framework. It provides a Python-first experience for training and inference. Mugen is a C++ inference engine with hand-tuned Metal kernels specifically optimized for MoE architectures and SSD offloading — a narrower scope with deeper optimization in that niche.

---

## Model Compatibility

### What model formats does Mugen support?

Mugen reads standard **GGUF** files directly. No conversion is needed if you already have GGUF models (the same files used by llama.cpp).

### What quantization formats are supported?

Q4_0, Q4_K, Q8_0, and F16. Each has a corresponding fused dequant + GEMV Metal kernel. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the full matrix.

### Can I use safetensors or PyTorch models?

Not directly. Convert them to GGUF first using the `llama.cpp` conversion tools (`convert_hf_to_gguf.py`). Mugen reads GGUF natively.

### Which MoE architectures are supported?

Currently tested: DeepSeek V2-Lite, OLMoE (64 experts), and dense models up to 7B. DeepSeek V3 (671B) is the North Star target and is under active development. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the full list.

---

## Performance

### What decode speed should I expect?

It depends on the model, quantization, and your hardware:

| Model | Hardware | Decode |
|-------|----------|--------|
| Llama-3.2-1B Q4_0 | M4 Pro 64 GB | 266 tok/s |
| Qwen2.5-7B Q4_0 | M4 Pro 64 GB | 56 tok/s |
| DeepSeek-V2-Lite Q8_0 | M4 Pro 64 GB | 4.9 tok/s (CPU MLA, GPU WIP) |

MoE models that use SSD offload will be slower than fully resident models — the goal is feasibility, not parity with in-RAM inference.

### How do I improve performance?

See [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) for detailed tuning advice. Quick wins:
1. Use Q4_K or Q4_0 quantization to reduce model size
2. Close other applications to free unified memory
3. Disable macOS swap (`sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist`)
4. Ensure SSD TRIM is enabled (`sudo trimforce enable`)

---

## Troubleshooting

### I get "out of memory" errors

Mugen tries to keep frequently-used expert weights in Metal buffers. If you are running other memory-intensive applications, Mugen's LRU eviction may thrash. Solutions:
1. Close other apps, especially browsers with many tabs
2. Use a more aggressively quantized model (Q4_0 instead of Q8_0)
3. Run `mugen-cli doctor` to check available memory
4. For MoE models, ensure USPP is enabled so experts stream from SSD on demand

### Build fails with Metal shader errors

Ensure Xcode is installed (not just Command Line Tools) and is selected:
```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

### `mugen-cli doctor` reports warnings

This tool checks your system for optimal Mugen performance. Address any warnings it reports — common issues include insufficient free memory, swap enabled, or SSD TRIM disabled.

---

## Why Not Linux / NVIDIA?

### Will Mugen support NVIDIA GPUs or Linux?

Not in the foreseeable future. Mugen is designed around Apple Silicon's **unified memory architecture**, where CPU and GPU share the same physical memory with zero-copy access. This architectural property is fundamental to how USPP works — memory-mapped GGUF files can be promoted to Metal buffers without copying data between CPU and GPU address spaces.

NVIDIA's discrete GPU architecture requires explicit CPU → GPU transfers over PCIe, which would negate USPP's zero-copy design. Supporting CUDA would require a fundamentally different approach, not just a backend swap.

If you need NVIDIA/Linux support, [llama.cpp](https://github.com/ggerganov/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm) are excellent choices.

---

## Contributing

### How can I contribute?

Read [CONTRIBUTING.md](../CONTRIBUTING.md) for coding style, PR guidelines, and development setup. High-impact areas:
- GPU MLA attention kernel (the top priority for DeepSeek V2/V3 performance)
- Additional quantization format kernels
- Benchmarking on different Apple Silicon chips
- Documentation improvements

### I found a bug — where do I report it?

Open a [GitHub issue](https://github.com/Zaoqu-Liu/Mugen/issues) with:
1. Output of `mugen-cli doctor`
2. The exact command you ran
3. Model file name and quantization
4. Full error output
