#!/usr/bin/env bash
#
# Start vLLM serving Qwen3-30B-A3B-Instruct on 1x H100 80GB.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload profile this config targets (drives every flag below):
#   - prompts ~1.5-3K tokens (schema + system prompt dominate, and repeat)
#   - short structured SQL outputs (tens-to-low-hundreds of tokens)
#   - 2-3 dependent LLM calls per user question (generate -> verify -> revise)
#   - SLO: p95 end-to-end agent latency < 5s, >= 10 RPS
#
# Qwen3-30B-A3B is a Mixture-of-Experts model: 30B total params but only ~3B
# active per token. Compute/decode is cheap; the binding constraint on one
# H100 is KV-cache memory, so most flags below buy KV headroom or reuse.
#
# ENVIRONMENT NOTES (learned the hard way on the H100 run):
#  * The pinned vllm==0.10.2 in uv.lock CRASHES loading this model's tokenizer
#    ("Qwen2Tokenizer has no attribute all_special_tokens_extended"). vLLM 0.11+
#    rewrote that code path, so we upgrade below. Because `uv run` re-syncs the
#    venv back to the lockfile, we install into the venv and launch the venv
#    python DIRECTLY (no `uv run`) so the upgrade sticks.
#  * VLLM_USE_FLASHINFER_SAMPLER=0 avoids FlashInfer's JIT sampler kernel, which
#    needs `ninja`+`nvcc` to build on first run. We decode greedily (temp=0), so
#    the built-in Torch sampler is equivalent here.

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# Ensure a vLLM that can load this tokenizer (no-op if already satisfied).
uv pip install -U "vllm>=0.11" "transformers" >/dev/null

# Use built-in Torch sampler; skip FlashInfer JIT (no ninja/nvcc needed).
export VLLM_USE_FLASHINFER_SAMPLER=0

exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    `# Single H100 -> no tensor/pipeline parallelism.` \
    --tensor-parallel-size 1 \
    `# Prompts top out ~3K tokens; capping at 8192 (vs the model default of` \
    `# 256K) shrinks the per-sequence KV reservation massively -> far more` \
    `# concurrent requests fit, which is what the 10 RPS target needs.` \
    --max-model-len 8192 \
    `# Use as much of the 80GB card as is safe for weights + KV cache.` \
    --gpu-memory-utilization 0.90 \
    `# FP8 KV cache halves KV memory vs fp16 -> ~2x concurrency headroom on` \
    `# the H100 (native FP8). Accuracy impact on short SQL outputs is negligible.` \
    --kv-cache-dtype fp8 \
    `# The schema + system prompt are identical across the 2-3 calls within a` \
    `# request AND across every question hitting the same DB. Prefix caching` \
    `# skips re-prefilling that shared 1.5-3K-token prefix -> big TTFT + KV win.` \
    --enable-prefix-caching \
    `# Concurrency cap: high enough to absorb the agent's bursty dependent calls` \
    `# at 10 RPS. Tune in Phase 6 against KV-cache usage / preemption metrics.` \
    --max-num-seqs 64

# NOTE: chunked prefill is ON by default in vLLM 0.11+ (no flag needed). The old
# --disable-log-requests flag was removed after 0.12, so it is dropped here too.
#
# --- Phase 6 levers to reach for if the dashboard says so -----------------
#  * KV cache pegged at ~1.0 + preemptions > 0  -> serve an FP8 *weights*
#    variant (e.g. .../Qwen3-30B-A3B-Instruct-2507-FP8) to free ~30GB for KV,
#    or lower --max-model-len further, or raise --gpu-memory-utilization to 0.93.
#  * waiting queue grows while running is flat   -> raise --max-num-seqs.
#  * TTFT spikes under load                      -> lower --max-num-batched-tokens
#    (e.g. 4096) to shrink prefill chunks and protect decode latency.
