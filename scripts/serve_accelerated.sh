#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
backend="${1:?Usage: bash scripts/serve_accelerated.sh vllm|sglang [extra server arguments]}"
shift
model="${MODEL_PATH:-${repo_dir}/pretrained_models/FireRedAudio-backbone}"
host="${FIREREDAUDIO_HOST:-127.0.0.1}"
port="${PORT:-8010}"
tp="${TENSOR_PARALLEL_SIZE:-1}"
max_len="${MAX_MODEL_LEN:-4096}"

case "$backend" in
  vllm)
    engine_python="${ENGINE_PYTHON:-${repo_dir}/.venv-vllm/bin/python}"
    export PATH="$(dirname -- "$engine_python"):$PATH"
    exec "$engine_python" -m fireredaudio.accelerated.serve_vllm \
      --model "$model" --served-model-name fireredaudio --host "$host" --port "$port" \
      --dtype bfloat16 --tensor-parallel-size "$tp" --max-model-len "$max_len" \
      --mamba-ssm-cache-dtype float32 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
      --max-num-seqs "${MAX_NUM_SEQS:-8}" --enable-prompt-embeds \
      --no-enable-prefix-caching --enforce-eager "$@"
    ;;
  sglang)
    export SGLANG_EXTERNAL_MODEL_PACKAGE=fireredaudio.accelerated.sglang_models
    # This text-only wrapper has no Conv3d; the cuDNN vision-model check is irrelevant.
    export SGLANG_DISABLE_CUDNN_CHECK=1
    engine_python="${ENGINE_PYTHON:-${repo_dir}/.venv-sglang/bin/python}"
    export PATH="$(dirname -- "$engine_python"):$PATH"
    exec "$engine_python" -m fireredaudio.accelerated.serve_sglang \
      --model-path "$model/sglang" --served-model-name fireredaudio --host "$host" --port "$port" \
      --dtype bfloat16 --tp-size "$tp" --context-length "$max_len" \
      --mem-fraction-static "${GPU_MEMORY_UTILIZATION:-0.75}" \
      --max-running-requests "${MAX_NUM_SEQS:-8}" --disable-radix-cache \
      --chunked-prefill-size -1 --disable-cuda-graph --attention-backend triton \
      --random-seed "${SEED:-42}" "$@"
    ;;
  *) printf 'Unknown backend: %s\n' "$backend" >&2; exit 2 ;;
esac
