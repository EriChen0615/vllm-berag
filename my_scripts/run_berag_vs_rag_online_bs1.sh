#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source my_scripts/activate_env.sh >/dev/null

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
DATA_DIR="${DATA_DIR:-my_outputs/data/NarrativeQA}"
QUERY_IMAGE_PATH="${QUERY_IMAGE_PATH:-my_data/just-a-random-picture.webp}"
QUERY_IMAGE_UUID="${QUERY_IMAGE_UUID:-narrativeqa-shared-query-image}"
K_VALUES="${K_VALUES:-50,100,150,200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-256}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-auto}"
NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"
PRIOR_MODE="${PRIOR_MODE:-uniform}"
DEFAULT_PRIOR_TOKEN_OFFSET="${DEFAULT_PRIOR_TOKEN_OFFSET:--4}"
RUN_ORDER="${RUN_ORDER:-rag-first}"
SUMMARY_SKIP_FIRST_ROWS="${SUMMARY_SKIP_FIRST_ROWS:-1}"

cmd=(
  .venv/bin/python my_scripts/benchmark_berag_vs_rag_bs1.py
  --model "$MODEL"
  --data-dir "$DATA_DIR"
  --k-values "$K_VALUES"
  --max-examples "$MAX_EXAMPLES"
  --request-batch-size "$REQUEST_BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
  --query-image-path "$QUERY_IMAGE_PATH"
  --query-image-uuid "$QUERY_IMAGE_UUID"
  --num-accumulator-rows "$NUM_ACCUMULATOR_ROWS"
  --pruning-top-p "$PRUNING_TOP_P"
  --prior-mode "$PRIOR_MODE"
  --default-prior-token-offset "$DEFAULT_PRIOR_TOKEN_OFFSET"
  --run-order "$RUN_ORDER"
  --summary-skip-first-rows "$SUMMARY_SKIP_FIRST_ROWS"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  cmd+=(--trust-remote-code)
else
  cmd+=(--no-trust-remote-code)
fi

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "${DISABLE_TQDM:-0}" == "1" ]]; then
  cmd+=(--disable-tqdm)
fi

if [[ "${STOP_ON_ERROR:-1}" == "1" ]]; then
  cmd+=(--stop-on-error)
fi

if [[ "${NO_RAG_TRUNCATION:-0}" == "1" ]]; then
  cmd+=(--no-rag-truncation)
else
  cmd+=(--rag-truncation-side "${RAG_TRUNCATION_SIDE:-right}")
  if [[ -n "${RAG_TRUNCATE_PROMPT_TOKENS:-}" ]]; then
    cmd+=(--rag-truncate-prompt-tokens "$RAG_TRUNCATE_PROMPT_TOKENS")
  fi
fi

if [[ "${BERAG_LOG_GROUPS:-0}" == "1" ]]; then
  cmd+=(--berag-log-groups)
fi

if [[ "${BERAG_LOG_FULL_POSTERIOR:-0}" == "1" ]]; then
  cmd+=(--berag-log-full-posterior)
fi

if [[ "${DEBUG:-0}" == "1" ]]; then
  cmd+=(--debug)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[compare-bs1] repo=$REPO_ROOT"
echo "[compare-bs1] model=$MODEL"
echo "[compare-bs1] k_values=$K_VALUES"
echo "[compare-bs1] max_examples=$MAX_EXAMPLES"
echo "[compare-bs1] max_tokens=$MAX_TOKENS"
echo "[compare-bs1] max_model_len=$MAX_MODEL_LEN"
echo "[compare-bs1] max_num_seqs=$MAX_NUM_SEQS"
echo "[compare-bs1] query_image_path=$QUERY_IMAGE_PATH"
echo "[compare-bs1] command: ${cmd[*]}"
"${cmd[@]}"
