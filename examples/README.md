# Mugen Examples

Hands-on guides for common Mugen workflows. Each example is self-contained — pick the one that matches your use case and follow along.

| Example | Description |
|---------|-------------|
| [basic_chat](basic_chat/) | Interactive chat with a GGUF model using `mugen-cli` |
| [speculative_decoding](speculative_decoding/) | Accelerate generation with draft-model speculative decoding |
| [api_server](api_server/) | Launch an OpenAI-compatible HTTP server and query it from Python or cURL |
| [benchmarking](benchmarking/) | Run, interpret, and compare performance benchmarks |
| [moe_inference](moe_inference/) | Run Mixture-of-Experts models with USPP pipelining on Apple Silicon |

## Prerequisites

- macOS 14+ on Apple Silicon (M1 or later)
- Mugen installed and `mugen-cli` on your `PATH` (see the [main README](../README.md))
- At least 16 GB unified memory (32 GB+ recommended for large models)

## Downloading Models

All examples use publicly available GGUF models from Hugging Face. Download with the Hugging Face CLI or any HTTP client:

```console
# Install the Hugging Face CLI (one-time)
$ pip install huggingface-hub

# Download a model
$ huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
    llama-2-7b-chat.Q4_K_M.gguf \
    --local-dir ./models
```

## Conventions

- Terminal sessions use `$` for the shell prompt and omit it for output lines.
- All file sizes and throughput numbers are representative; your results will vary with hardware and quantization.
- Python scripts target Python 3.10+ and list dependencies at the top of each file.
