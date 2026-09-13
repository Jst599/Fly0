#!/usr/bin/env bash
# ============================================================
# LoRA 微调 Qwen3-VL-4B（视觉指代定位 grounding）
# 在 AutoDL 的 conda 环境 swift 里运行:  bash train_qwen3vl.sh
#
# 数据来源默认是 prepare_subset.py 生成的 train.jsonl / val.jsonl（最稳）。
# 若你的 ms-swift 支持内置数据集切片（swift sft --help 确认），
# 把下面两个变量改成 'swift/refcoco#20000' 和 'swift/refcoco#2000' 即可。
# ============================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TRAIN_DATASET="${TRAIN_DATASET:-train.jsonl}"
VAL_DATASET="${VAL_DATASET:-val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3vl-4b-refcoco-lora}"

echo "==> model:    $MODEL"
echo "==> dataset:  $TRAIN_DATASET / $VAL_DATASET"
echo "==> output:   $OUTPUT_DIR"

# 跑前核对参数名（版本不同可能略异）:
#   swift sft --help | grep -E "target_modules|freeze_vit|freeze_aligner|merge_lora|train_type|lora_rank"
swift sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$TRAIN_DATASET" \
    --val_dataset "$VAL_DATASET" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --merge_lora true
