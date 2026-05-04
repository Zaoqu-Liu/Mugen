# Mugen 北极星

> 在这台 Mac mini M4 Pro (64GB) 上跑全量 DeepSeek V3。
> 
> 671B 参数。MoE。Q4 量化。从 SSD 流式加载。投机解码隐藏延迟。
> 
> 不是 demo，是能用的。

## 这意味着什么

- 671B Q4 ≈ 340 GB 权重。64GB 内存装不下。**必须 SSD offload**。
- 256 experts / 8 active。每 token 需要 8 个 expert 的权重。**必须预测下一步要哪些 expert，提前从 SSD 预取**。
- SSD 读取延迟 ≈ 0.1ms/page，但 8 个 expert × ~20MB/expert = 160MB/token。**必须用投机解码生成的 draft 序列来并行预取多步 expert**。
- 目标 decode ≥ 8 tok/s。用户体验的底线。

## 当前到北极星的距离

```
[已完成] Phase 1 核心引擎           ████████████████████ 100%
  - Dense 推理 (0.5B/3B/7B)        ✅
  - MoE 推理 (OLMoE 64 experts)    ✅
  - Flash Attention (Prefill+Decode) ✅
  - 跨层 CB Batching (7B 33 tok/s) ✅
  - KV Cache 复用                   ✅

[已完成] Phase 1.5 MoE 性能         ████████████████████ 100%
  - MoE batch routing (T++)         ✅ OLMoE 12.6→130.7 tok/s (10.4x)
  - scatter_kv GPU kernel (L)       ✅ 25th kernel, 代码架构改进
  - Llama 3 架构支持 (ARCH)         ✅ 126ms TTFT / 110 tok/s decode

[已完成] Phase 2 USPP 实战          ████████████████████ 100%
  - draft+target 投机解码端到端     ✅ 算法正确（greedy=纯7B），in-memory 无加速（需SSD）
  - Mega-chain decode 加速          ✅ 29→1 sync, 7B 32.9→39.6 tok/s (+20%)
  - SSD offload PoC                 ✅ page fault 测量 + 投机解码可行性验证
  - Router Logits API (WP-1)        ✅ TransformerModel 路由回调，MoE 三条路径
  - BufferManager 对接 (WP-3)       ✅ find_expert → Metal staging → MoE forward 消费
  - Route Predictor (WP-2)          ✅ draft→target 层映射 + expert clamp + 优先级排序
  - Async Prefetch Pipeline (WP-4)  ✅ N-deep I/O 队列 + decode_step GPU-IO 重叠
  - E2E 集成测试 + 布局验证         ✅ 6 checkpoint PASS, staging 布局三方一致
  - DeepSeek V3 Gap List            ✅ 12 维差异分析 + 6 新 kernel 需求

[进行中] Phase 3 DeepSeek V3        ████████░░░░░░░░░░░░ 40%
  - MLA Decode (absorbed attention)  ✅
  - MLA Prefill (batch + CPU attn)   ✅
  - Shared Expert (4 MoE paths)      ✅
  - RoutePredictor 分桶映射          ✅
  - 分组 MoE 路由 (sigmoid+group)    ✅
  - DS V2-Lite 端到端验证 (R²=0.998)  🟡 logits正确，chat输出待修
  - DS V3 GGUF 端到端验证              ⬜
  - 三层存储调度                     ⬜
  - 671B 权重 SSD offload            ⬜
  - USPP 投机-预取融合               ⬜
  - MLA GPU Attention Kernel         ⬜
  - ≥ 8 tok/s                        ⬜
```

## 每个 Agent 会话都要记住

1. **一切优化都服务于 671B on 64GB 这个终局**。不追求小模型上的极限数字。
2. **USPP 是 Mugen 存在的唯一理由**。没有 USPP，Mugen 就是功能更少的 llama.cpp。
3. **SSD offload 是 Phase 2-3 的核心**。Phase 1 的所有工作都是在为它打地基。
4. **MoE batch 优化不是可选项**。671B 有 256 experts，per-token fallback = 死。
5. **不要在小模型上过度优化**。0.5B 从 70→80 tok/s 不值得花时间。7B→DeepSeek 的路径才重要。
