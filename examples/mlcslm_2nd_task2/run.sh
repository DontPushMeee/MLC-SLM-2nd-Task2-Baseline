#!/bin/bash
# Script to train and evaluate the Qwen2.5-Omni-7B model with Megatron-LM.
# Stages:
#   0 - Convert model to Megatron format
#   1 - Train with Megatron
#   2 - Convert trained adapters to HuggingFace format
#   3 - Run inference on the dev set

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
stage=0
stop_stage=3

ROOT_PATH=${ROOT_PATH:-$(cd "$(dirname "$0")" && pwd)}
export TMP_PATH=$ROOT_PATH/.tmp
export PIP_CACHE_PATH=$ROOT_PATH/.cache
export MODELSCOPE_CACHE=$ROOT_PATH/.shared
export MEGATRON_LM_PATH=$ROOT_PATH/../../Megatron-LM

# Data paths
TRAIN_JSONL="$ROOT_PATH/datasets/train.jsonl"
VAL_JSONL="$ROOT_PATH/datasets/dev.jsonl"
TEST_JSONL="$ROOT_PATH/datasets/dev.jsonl"    # used for final inference

# Output paths
OUTPUT_DIR="$ROOT_PATH/outputs/train"
HF_EXPORT_DIR="$ROOT_PATH/output/train/train_hf"
INFER_SAVE_PATH="$HF_EXPORT_DIR.dev.jsonl"

# ----------------------------------------------------------------------
# Stage 0: Convert to Megatron format
# ----------------------------------------------------------------------
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Convert Qwen2.5-Omni-7B to Megatron format"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    swift export \
        --model "$ROOT_PATH/models/Qwen2.5-Omni-7B" \
        --to_mcore true \
        --torch_dtype bfloat16 \
        --output_dir "$ROOT_PATH/models/Qwen2.5-Omni-7B-mcore"
fi

# ----------------------------------------------------------------------
# Stage 1: Megatron training (LoRA)
# ----------------------------------------------------------------------
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Start Megatron-LM training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NPROC_PER_NODE=8 \
    MAX_PIXELS=1003520 \
    VIDEO_MAX_PIXELS=50176 \
    FPS_MAX_FRAMES=12 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    megatron sft \
        --load "$ROOT_PATH/models/Qwen2.5-Omni-7B-mcore" \
        --dataset "$TRAIN_JSONL" \
        --val_dataset "$VAL_JSONL" \
        --load_from_cache_file true \
        --train_type lora \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --tensor_model_parallel_size 4 \
        --pipeline_model_parallel_size 2 \
        --context_parallel_size 1 \
        --sequence_parallel true \
        --packing true \
        --freeze_llm false \
        --freeze_vit false \
        --freeze_aligner false \
        --micro_batch_size 1 \
        --global_batch_size 32 \
        --recompute_granularity full \
        --recompute_method uniform \
        --recompute_num_layers 1 \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --lr 1e-5 \
        --lr_warmup_fraction 0.05 \
        --min_lr 1e-6 \
        --max_epochs 100 \
        --save "$OUTPUT_DIR" \
        --eval_interval 200 \
        --save_interval 200 \
        --vit_gradient_checkpointing true \
        --max_length 98304 \
        --num_workers 16 \
        --dataset_num_proc 16 \
        --no_save_optim true \
        --no_save_rng true \
        --attention_backend flash
fi

# ----------------------------------------------------------------------
# Stage 2: Convert trained adapters to HuggingFace format
# ----------------------------------------------------------------------
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Convert Megatron adapters to HuggingFace format"

    # Find the latest training output directory (there will be only one)
    train_dir=$(ls -dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n 1 || true)
    if [ -z "$train_dir" ]; then
        echo "Error: no training output found in $OUTPUT_DIR"
        exit 1
    fi
    mcore_adapters="${train_dir%/}"   # remove trailing slash
    log "Using adapter: $mcore_adapters"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    swift export \
        --mcore_adapters "$mcore_adapters" \
        --to_hf true \
        --torch_dtype bfloat16 \
        --output_dir "$HF_EXPORT_DIR"
fi

# ----------------------------------------------------------------------
# Stage 3: Inference on dev/test set
# ----------------------------------------------------------------------
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Run inference with train_infer.py"
    python ./train_infer.py \
        --model_path "$HF_EXPORT_DIR" \
        --jsonl_path "$TEST_JSONL" \
        --save_path "$INFER_SAVE_PATH"
fi

log "All stages finished."