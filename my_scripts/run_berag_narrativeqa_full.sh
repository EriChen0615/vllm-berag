#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source my_scripts/activate_env.sh >/dev/null

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-my_outputs/data/NarrativeQA}"
QUERY_IMAGE_PATH="${QUERY_IMAGE_PATH:-}"
QUERY_IMAGE_UUID="${QUERY_IMAGE_UUID:-narrativeqa-shared-query-image}"
# K_VALUES="${K_VALUES:-50,75,100,150,200}"
K_VALUES="${K_VALUES:-50,75,100}"
MAX_EXAMPLES="${MAX_EXAMPLES:-512}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-}"
MAX_TOKENS="${MAX_TOKENS:-32}"
# MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-auto}"
NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"
# PRUNING_TOP_P="${PRUNING_TOP_P:-0.9}"
PRIOR_MODE="${PRIOR_MODE:-uniform}"
DEFAULT_PRIOR_TOKEN_OFFSET="${DEFAULT_PRIOR_TOKEN_OFFSET:--4}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL//\//_}}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
EXP_NAME="${EXP_NAME:-berag_${PRIOR_MODE}_${MODEL_SLUG}_TOPP-${PRUNING_TOP_P}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-my_outputs/experiments/${EXP_NAME}}"

echo "[berag] repo=$REPO_ROOT"
echo "[berag] model=$MODEL"
echo "[berag] data_dir=$DATA_DIR"
echo "[berag] query_image_path=${QUERY_IMAGE_PATH:-none}"
echo "[berag] output_dir=$OUTPUT_DIR"
echo "[berag] k_values=$K_VALUES"
echo "[berag] max_examples=$MAX_EXAMPLES"
echo "[berag] request_batch_size=${REQUEST_BATCH_SIZE:-default}"
echo "[berag] max_tokens=$MAX_TOKENS"
echo "[berag] max_model_len=$MAX_MODEL_LEN"
echo "[berag] max_num_seqs=$MAX_NUM_SEQS"
echo "[berag] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "[berag] num_accumulator_rows=$NUM_ACCUMULATOR_ROWS"
echo "[berag] pruning_top_p=$PRUNING_TOP_P"
echo "[berag] prior_mode=$PRIOR_MODE"

.venv/bin/python my_scripts/validate_narrativeqa_data.py \
  --data-dir "$DATA_DIR" \
  --k-values "$K_VALUES" \
  --max-examples "$MAX_EXAMPLES"

IFS=',' read -ra K_ARRAY <<< "$K_VALUES"
for K_VALUE_RAW in "${K_ARRAY[@]}"; do
  K_VALUE="${K_VALUE_RAW//[[:space:]]/}"
  if [[ -z "$K_VALUE" ]]; then
    continue
  fi

  RUN_NUM_ACCUMULATOR_ROWS="$NUM_ACCUMULATOR_ROWS"

  cmd=(
    .venv/bin/python my_scripts/benchmark_berag_narrativeqa.py
    --model "$MODEL"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --k-values "$K_VALUE"
    --max-examples "$MAX_EXAMPLES"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --num-accumulator-rows "$RUN_NUM_ACCUMULATOR_ROWS"
    --pruning-top-p "$PRUNING_TOP_P"
    --prior-mode "$PRIOR_MODE"
    --default-prior-token-offset "$DEFAULT_PRIOR_TOKEN_OFFSET"
    --trust-remote-code
    # --enforce-eager
  )

  if [[ "$PRIOR_MODE" == "module" ]]; then
    if [[ -n "${PRIOR_MODULE_CLS:-}" ]]; then
      cmd+=(--prior-module-cls "$PRIOR_MODULE_CLS")
    fi
    if [[ -n "${PRIOR_MODULE_WEIGHTS_PATH:-}" ]]; then
      cmd+=(--prior-module-weights-path "$PRIOR_MODULE_WEIGHTS_PATH")
    fi
    if [[ -n "${PRIOR_HIDDEN_SIZE:-}" ]]; then
      cmd+=(--prior-hidden-size "$PRIOR_HIDDEN_SIZE")
    fi
  fi

  if [[ -n "$QUERY_IMAGE_PATH" ]]; then
    cmd+=(
      --query-image-path "$QUERY_IMAGE_PATH"
      --query-image-uuid "$QUERY_IMAGE_UUID"
    )
  fi

  if [[ -n "${MAX_NUM_SEQS:-4096}" ]]; then
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
  fi

  if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi

  if [[ -n "$REQUEST_BATCH_SIZE" ]]; then
    cmd+=(--request-batch-size "$REQUEST_BATCH_SIZE")
  fi

  if [[ "${BERAG_LOG_GROUPS:-0}" == "1" ]]; then
    cmd+=(
      --berag-log-groups
      --berag-group-trace-path "$OUTPUT_DIR/berag/k${K_VALUE}/group_trace.jsonl"
    )
  fi

  if [[ "${BERAG_LOG_FULL_POSTERIOR:-0}" == "1" ]]; then
    cmd+=(--berag-log-full-posterior)
  fi

  if [[ "${DISABLE_TQDM:-0}" == "1" ]]; then
    cmd+=(--disable-tqdm)
  fi

  if [[ "${DEBUG:-0}" == "1" ]]; then
    cmd+=(--debug)
  fi

  if [[ "${STOP_ON_ERROR:-0}" == "1" ]]; then
    cmd+=(--stop-on-error)
  fi

  echo "[berag] starting k=$K_VALUE"
  echo "[berag] k=$K_VALUE num_accumulator_rows=$RUN_NUM_ACCUMULATOR_ROWS"
  "${cmd[@]}"
done

echo "[berag] done"
echo "[berag] results: $OUTPUT_DIR/berag"
