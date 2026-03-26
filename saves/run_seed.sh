#!/usr/bin/env bash
# Usage: bash run_seed.sh <seed> <gpu_ids>  e.g. bash run_seed.sh 0 "0,1,2,3"
set -euo pipefail

SEED=$1
GPUS=$2
VENV="/home/ebrahim/MM-NeuroOnco/.venv"
LF="/home/ebrahim/LLaMA-Factory"
PROJ="/home/ebrahim/MM-NeuroOnco"

TRAIN_OUT="$PROJ/saves/seed_${SEED}"
MERGED_OUT="$PROJ/saves/seed_${SEED}_merged"
STATS_OUT="$PROJ/results/eval_seed_${SEED}_stats.json"
PREDS_OUT="$PROJ/results/eval_seed_${SEED}_preds.jsonl"
LOG="$PROJ/saves/seed_${SEED}_train.log"

echo "===== SEED $SEED on GPUs $GPUS ====="

# --- Train ---
echo "[$(date)] Training seed $SEED..."
CUDA_VISIBLE_DEVICES=$GPUS uv run --python $VENV \
  python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=$((29600 + SEED)) \
    $LF/src/train.py \
      --stage sft \
      --do_train \
      --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
      --dataset neuroonco_nocot \
      --dataset_dir $LF/data \
      --template qwen2_vl \
      --finetuning_type lora \
      --lora_target q_proj,v_proj \
      --output_dir $TRAIN_OUT \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --save_steps 500 \
      --learning_rate 1e-4 \
      --num_train_epochs 1.0 \
      --bf16 \
      --ddp_find_unused_parameters False \
      --warmup_ratio 0.05 \
      --trust_remote_code True \
      --report_to none \
      --seed $SEED \
  > $LOG 2>&1
echo "[$(date)] Training done."

# --- Merge ---
echo "[$(date)] Merging LoRA..."
uv run --python $VENV python $PROJ/saves/merge_lora.py \
  --adapter $TRAIN_OUT \
  --out $MERGED_OUT \
  >> $LOG 2>&1
echo "[$(date)] Merge done."

# --- Eval ---
echo "[$(date)] Evaluating..."
CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run --python $VENV \
  python $PROJ/instruction_data/qwen3_vl_eval.py \
    --model $MERGED_OUT \
    --json_path $PROJ/Benchmark/Benchmark_VQA_Closed.json \
    --data_root $PROJ/images/Benchmark_Images \
    --output_stats $STATS_OUT \
    --output_preds $PREDS_OUT \
  >> $LOG 2>&1
echo "[$(date)] Eval done. Results: $STATS_OUT"
