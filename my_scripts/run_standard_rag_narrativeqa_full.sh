#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source my_scripts/activate_env.sh >/dev/null

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-my_outputs/data/NarrativeQA}"
K_VALUES="${K_VALUES:-50,75,100,150,200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-512}"
MAX_TOKENS="${MAX_TOKENS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TRUNCATE_PROMPT_TOKENS="${TRUNCATE_PROMPT_TOKENS:-$((MAX_MODEL_LEN - MAX_TOKENS))}"
TRUNCATION_SIDE="${TRUNCATION_SIDE:-right}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-auto}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL//\//_}}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
EXP_NAME="${EXP_NAME:-standard_rag_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-my_outputs/experiments/${EXP_NAME}}"

echo "[standard-rag] repo=$REPO_ROOT"
echo "[standard-rag] model=$MODEL"
echo "[standard-rag] data_dir=$DATA_DIR"
echo "[standard-rag] output_dir=$OUTPUT_DIR"
echo "[standard-rag] k_values=$K_VALUES"
echo "[standard-rag] max_examples=$MAX_EXAMPLES"
echo "[standard-rag] max_tokens=$MAX_TOKENS"
echo "[standard-rag] max_model_len=$MAX_MODEL_LEN"
echo "[standard-rag] truncate_prompt_tokens=$TRUNCATE_PROMPT_TOKENS"
echo "[standard-rag] truncation_side=$TRUNCATION_SIDE"
echo "[standard-rag] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"

.venv/bin/python my_scripts/validate_narrativeqa_data.py \
  --data-dir "$DATA_DIR" \
  --k-values "$K_VALUES" \
  --max-examples "$MAX_EXAMPLES"

cmd=(
  .venv/bin/python my_scripts/benchmark_standard_rag_narrativeqa.py
  --model "$MODEL"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --k-values "$K_VALUES"
  --max-examples "$MAX_EXAMPLES"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --truncate-prompt-tokens "$TRUNCATE_PROMPT_TOKENS"
  --truncation-side "$TRUNCATION_SIDE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
  --enforce-eager
  --trust-remote-code
)

if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${DISABLE_TQDM:-0}" == "1" ]]; then
  cmd+=(--disable-tqdm)
fi

echo "[standard-rag] starting benchmark"
"${cmd[@]}"

echo "[standard-rag] done"
echo "[standard-rag] results: $OUTPUT_DIR/standard_rag"
