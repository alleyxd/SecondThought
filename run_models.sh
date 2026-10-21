#!/usr/bin/env bash
# Runs every condition for each model: start a vLLM container with tool calling, wait until it serves, run, stop.
# Usage: bash run_models.sh "Qwen/Qwen3-8B:qwen3-8b" "Qwen/Qwen3-4B:qwen3-4b"
set -u
cd "$(dirname "$0")"
HF="C:/Users/alias/.cache/huggingface"
for spec in "$@"; do
  model="${spec%%:*}"; label="${spec##*:}"
  echo "=== $label ($model) $(date +%H:%M)"
  docker rm -f toolcall-llm >/dev/null 2>&1
  docker run -d --name toolcall-llm --gpus all --ipc=host -p 8000:8000 -v "$HF:/root/.cache/huggingface" \
    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm/vllm-openai:latest \
    --model "$model" --served-model-name local --gpu-memory-utilization 0.65 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 >/dev/null
  for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -s localhost:8000/v1/models >/dev/null || { echo "server for $label did not start"; docker logs --tail 20 toolcall-llm; exit 1; }
  PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m toolcall.run --label "$label" --batch 32 --duty 0.5 2>&1
  docker rm -f toolcall-llm >/dev/null 2>&1
done
echo "all models finished"
