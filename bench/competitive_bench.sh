#!/bin/bash
# Mugen Competitive Benchmark — 三方对比 (Mugen vs ollama vs MLX)
# 每次性能优化后必须跑此脚本，结果写入 bench/competitive_results.md
#
# 用法: bash bench/competitive_bench.sh
# 前置: ollama 已安装 + ollama serve 运行中 + mlx-lm 已安装 + Mugen 已构建

set -e

RESULTS_FILE="bench/competitive_results.md"
MODELS_DIR="$HOME/.mugen/models"
SLEEP_BETWEEN=8  # 避免热节流

echo "=== Mugen Competitive Benchmark ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ─── 1. Mugen Benchmark ───
echo ">>> [1/3] Mugen 7B Q4_0 benchmark..."
MUGEN_7B=$(./build/src/mugen-cli bench "$MODELS_DIR/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf" 2>&1)
MUGEN_7B_DECODE=$(echo "$MUGEN_7B" | grep -o '"tokens_per_sec": [0-9.]*' | tail -1 | grep -o '[0-9.]*')
MUGEN_7B_PREFILL=$(echo "$MUGEN_7B" | grep -o '"tokens_per_sec": [0-9.]*' | head -1 | grep -o '[0-9.]*')
MUGEN_7B_TTFT=$(echo "$MUGEN_7B" | grep -o '"time_ms": [0-9.]*' | head -1 | grep -o '[0-9.]*')
echo "  Mugen 7B: decode=${MUGEN_7B_DECODE} tok/s, prefill=${MUGEN_7B_PREFILL} tok/s, TTFT=${MUGEN_7B_TTFT}ms"

sleep $SLEEP_BETWEEN

echo ">>> [1/3] Mugen 1B benchmark..."
MUGEN_1B=$(./build/src/mugen-cli bench "$MODELS_DIR/llama-3.2-1b-instruct-q4_0.gguf" 2>&1)
MUGEN_1B_DECODE=$(echo "$MUGEN_1B" | grep -o '"tokens_per_sec": [0-9.]*' | tail -1 | grep -o '[0-9.]*')
echo "  Mugen 1B: decode=${MUGEN_1B_DECODE} tok/s"

sleep $SLEEP_BETWEEN

# ─── 2. ollama Benchmark ───
echo ">>> [2/3] ollama 7B Q4_0 benchmark..."
echo "  Pulling model if needed..."
ollama pull qwen2.5:7b-instruct-q4_0 2>&1 | tail -1

# Warmup
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b-instruct-q4_0","prompt":"hi","stream":false,"options":{"num_predict":5}}' > /dev/null 2>&1

sleep 3

# Benchmark: 类似 Mugen 的 prompt 长度 + 128 gen tokens
OLLAMA_7B=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct-q4_0",
  "prompt": "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes. Include mathematical formulations where appropriate.",
  "stream": false,
  "options": {"num_predict": 128, "temperature": 0}
}')

OLLAMA_7B_TOTAL_NS=$(echo "$OLLAMA_7B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_duration',0))" 2>/dev/null || echo "0")
OLLAMA_7B_PROMPT_COUNT=$(echo "$OLLAMA_7B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo "0")
OLLAMA_7B_PROMPT_NS=$(echo "$OLLAMA_7B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_duration',0))" 2>/dev/null || echo "0")
OLLAMA_7B_EVAL_COUNT=$(echo "$OLLAMA_7B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
OLLAMA_7B_EVAL_NS=$(echo "$OLLAMA_7B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo "0")

if [ "$OLLAMA_7B_EVAL_NS" -gt 0 ] 2>/dev/null; then
  OLLAMA_7B_DECODE=$(python3 -c "print(f'{$OLLAMA_7B_EVAL_COUNT / ($OLLAMA_7B_EVAL_NS / 1e9):.1f}')")
  OLLAMA_7B_PREFILL=$(python3 -c "print(f'{$OLLAMA_7B_PROMPT_COUNT / ($OLLAMA_7B_PROMPT_NS / 1e9):.1f}')" 2>/dev/null || echo "N/A")
  OLLAMA_7B_TTFT=$(python3 -c "print(f'{$OLLAMA_7B_PROMPT_NS / 1e6:.1f}')" 2>/dev/null || echo "N/A")
else
  OLLAMA_7B_DECODE="FAILED"
  OLLAMA_7B_PREFILL="FAILED"
  OLLAMA_7B_TTFT="FAILED"
fi
echo "  ollama 7B: decode=${OLLAMA_7B_DECODE} tok/s, prefill=${OLLAMA_7B_PREFILL} tok/s, TTFT=${OLLAMA_7B_TTFT}ms"

sleep $SLEEP_BETWEEN

# ─── 3. MLX Benchmark ───
echo ">>> [3/3] MLX 7B 4-bit benchmark..."
MLX_BENCH=$(python3 -c "
import time, sys
from mlx_lm import load, generate

model, tokenizer = load('mlx-community/Qwen2.5-7B-Instruct-4bit')

prompt = 'Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes. Include mathematical formulations where appropriate.'
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# warmup
generate(model, tokenizer, prompt=text, max_tokens=5, verbose=False)

# timed run
t0 = time.perf_counter()
response = generate(model, tokenizer, prompt=text, max_tokens=128, verbose=True)
t1 = time.perf_counter()
print(f'WALL_TIME_MS={((t1-t0)*1000):.1f}')
" 2>&1)

MLX_7B_TPS=$(echo "$MLX_BENCH" | grep -o 'Tokens/second.*: [0-9.]*' | tail -1 | grep -o '[0-9.]*$' || echo "N/A")
MLX_7B_PROMPT_TPS=$(echo "$MLX_BENCH" | grep -o 'Prompt.*Tokens-per-sec: [0-9.]*' | grep -o '[0-9.]*$' || echo "N/A")
MLX_7B_WALL=$(echo "$MLX_BENCH" | grep -o 'WALL_TIME_MS=[0-9.]*' | grep -o '[0-9.]*' || echo "N/A")

# Fallback: parse generation tokens-per-sec
if [ "$MLX_7B_TPS" = "N/A" ]; then
  MLX_7B_TPS=$(echo "$MLX_BENCH" | grep -oE '[0-9.]+ tokens-per-sec' | tail -1 | grep -o '^[0-9.]*' || echo "N/A")
fi

echo "  MLX 7B: decode=${MLX_7B_TPS} tok/s, wall=${MLX_7B_WALL}ms"

# ─── 生成报告 ───
echo ""
echo "=== Generating report ==="

cat > "$RESULTS_FILE" << REPORT_EOF
# Mugen Competitive Benchmark Results

> 测试时间: $(date '+%Y-%m-%d %H:%M:%S')
> 硬件: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon") / $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1073741824}') RAM
> 模型: Qwen2.5-7B-Instruct Q4

## 三方对比 (7B Q4, decode 128 tokens)

| 引擎 | 版本 | Decode tok/s | Prefill tok/s | TTFT |
|------|------|:---:|:---:|:---:|
| **Mugen** | 0.1.0 | **${MUGEN_7B_DECODE}** | ${MUGEN_7B_PREFILL} | ${MUGEN_7B_TTFT}ms |
| **ollama** | $(ollama --version 2>&1 | grep -o '[0-9.]*') | **${OLLAMA_7B_DECODE}** | ${OLLAMA_7B_PREFILL} | ${OLLAMA_7B_TTFT}ms |
| **MLX** | $(python3 -c "import mlx_lm; print(mlx_lm.__version__)" 2>/dev/null) | **${MLX_7B_TPS}** | ${MLX_7B_PROMPT_TPS:-N/A} | N/A |

## Mugen 全模型 (vs 自身历史)

| 模型 | Decode tok/s |
|------|:---:|
| Llama-3.2-1B Q4_0 | ${MUGEN_1B_DECODE} |
| Qwen2.5-7B Q4_0 | ${MUGEN_7B_DECODE} |

## Raw Output

<details>
<summary>Mugen 7B</summary>

\`\`\`
${MUGEN_7B}
\`\`\`
</details>

<details>
<summary>ollama 7B</summary>

\`\`\`json
${OLLAMA_7B}
\`\`\`
</details>

<details>
<summary>MLX 7B</summary>

\`\`\`
${MLX_BENCH}
\`\`\`
</details>
REPORT_EOF

echo ""
echo "✅ Report saved to: $RESULTS_FILE"
echo ""
echo "=== Summary ==="
echo "  Mugen  7B decode: ${MUGEN_7B_DECODE} tok/s"
echo "  ollama 7B decode: ${OLLAMA_7B_DECODE} tok/s"
echo "  MLX    7B decode: ${MLX_7B_TPS} tok/s"
echo "  Mugen  1B decode: ${MUGEN_1B_DECODE} tok/s"
