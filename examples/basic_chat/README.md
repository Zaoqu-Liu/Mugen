# Basic Chat

Run an interactive multi-turn conversation with a GGUF model on Apple Silicon.

## 1. Download a Model

```console
$ huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
    llama-2-7b-chat.Q4_K_M.gguf \
    --local-dir ./models
```

The Q4_K_M quantization strikes a good balance between quality and memory usage (~4.1 GB).

## 2. Start a Chat Session

```console
$ mugen-cli chat --model ./models/llama-2-7b-chat.Q4_K_M.gguf
```

You will see the model load and a prompt appear:

```console
[mugen] Loading llama-2-7b-chat.Q4_K_M.gguf … done (1.2 s)
[mugen] Metal backend: Apple M2 Max, 32 GB unified memory
[mugen] Context length: 4096 tokens

You> Hello! What can you help me with?

Assistant> I can help with a wide range of topics — writing, coding,
math, brainstorming, and general knowledge questions. What would you
like to work on?

You> Explain the Transformer architecture in three sentences.

Assistant> The Transformer processes input tokens in parallel through
layers of multi-head self-attention and feed-forward networks, removing
the sequential bottleneck of RNNs. Each attention head learns different
relationships between token positions, enabling the model to capture
both local and long-range dependencies. A final linear projection maps
hidden states to vocabulary logits for next-token prediction.

You> /quit
[mugen] Session ended. 47 tokens generated (62.3 tok/s).
```

## 3. Parameter Tuning

Control generation behavior with sampling flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Randomness. Lower → more deterministic. |
| `--top-p` | 0.9 | Nucleus sampling cutoff. |
| `--top-k` | 40 | Top-K sampling window. |
| `--max-tokens` | 2048 | Maximum tokens per response. |
| `--repeat-penalty` | 1.1 | Penalize repeated tokens. |
| `--ctx-size` | 4096 | Context window in tokens. |

### Precise, factual answers

```console
$ mugen-cli chat --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --temperature 0.2 --top-p 0.8 --max-tokens 512
```

### Creative, exploratory writing

```console
$ mugen-cli chat --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --temperature 0.9 --top-p 0.95 --max-tokens 4096
```

## 4. System Prompts

Provide a system prompt to steer the assistant persona:

```console
$ mugen-cli chat --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --system "You are a concise technical writer. Answer in bullet points."
```

## 5. Tips

- **Memory pressure.** If the system runs low on unified memory, use a smaller quantization (Q3_K_S) or reduce `--ctx-size`.
- **Chat templates.** Mugen auto-detects the chat template from GGUF metadata. Override with `--chat-template` if needed.
- **Logging.** Add `--verbose` to print per-token timing and Metal kernel dispatch info.
