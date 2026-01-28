#!/bin/bash
# Interactive Quick Training Script for Chain Image Restoration Framework
# Usage: bash training/quick_train.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHAIN_ROOT="$(dirname "$PROJECT_ROOT")"

# Default values (auto-detect based on project location)
DEFAULT_TRAIN_CONFIG="$CHAIN_ROOT/data/Comb_Config/Agent7/Agent7_Train_config.json"
DEFAULT_VAL_CONFIG="$CHAIN_ROOT/data/Comb_Config/Agent7/Agent7_Val_config.json"
DEFAULT_DEVICE="cuda:0"
DEFAULT_EPOCHS=15
DEFAULT_BATCH_SIZE=2
DEFAULT_GRADIENT_ACCUMULATION=2  # 有效batch_size = 2 * 2 = 4
DEFAULT_NUM_GPUS=1

# Change to project root
cd "$PROJECT_ROOT"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       Chain Quick Training - Interactive Setup${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to prompt with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local result

    echo -ne "${GREEN}$prompt${NC} [${YELLOW}$default${NC}]: "
    read -r result

    if [[ -z "$result" ]]; then
        echo "$default"
    else
        echo "$result"
    fi
}

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    local result

    if [[ "$default" == "y" ]]; then
        echo -ne "${GREEN}$prompt${NC} [${YELLOW}Y/n${NC}]: "
    else
        echo -ne "${GREEN}$prompt${NC} [${YELLOW}y/N${NC}]: "
    fi
    read -r result

    if [[ -z "$result" ]]; then
        result="$default"
    fi

    if [[ "$result" =~ ^[Yy] ]]; then
        echo "y"
    else
        echo "n"
    fi
}

# Function to validate file exists
validate_file() {
    local file="$1"
    local name="$2"

    if [[ ! -f "$file" ]]; then
        echo -e "${RED}Error: $name not found: $file${NC}"
        return 1
    fi
    return 0
}

# Interactive prompts
echo -e "${BLUE}>> Configuration Files${NC}"
echo ""
echo -e "${YELLOW}Note: Config files should be JSON format with 'pipelines' array${NC}"
echo -e "${YELLOW}      Each pipeline contains: id, pipeline (model list), data (lq/gt pairs)${NC}"
echo ""

echo -e "  ${GREEN}[Training Config]${NC} - Dataset configuration for training"
echo -e "  Default: ${YELLOW}$DEFAULT_TRAIN_CONFIG${NC}"
echo -ne "  Enter path (press Enter for default): "
read -r TRAIN_CONFIG
if [[ -z "$TRAIN_CONFIG" ]]; then
    TRAIN_CONFIG="$DEFAULT_TRAIN_CONFIG"
fi
if ! validate_file "$TRAIN_CONFIG" "Training config"; then
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Using: $TRAIN_CONFIG"
echo ""

echo -e "  ${GREEN}[Validation Config]${NC} - Dataset configuration for validation"
echo -e "  Default: ${YELLOW}$DEFAULT_VAL_CONFIG${NC}"
echo -ne "  Enter path (press Enter for default): "
read -r VAL_CONFIG
if [[ -z "$VAL_CONFIG" ]]; then
    VAL_CONFIG="$DEFAULT_VAL_CONFIG"
fi
if ! validate_file "$VAL_CONFIG" "Validation config"; then
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Using: $VAL_CONFIG"

echo ""
echo -e "${BLUE}>> Training Parameters${NC}"
echo ""
echo -e "${YELLOW}Note: Training uses progressive loss scheduling (L1 → Perceptual)${NC}"
echo -e "${YELLOW}      Recommended: 15+ epochs for best results${NC}"
echo ""

echo -e "  ${GREEN}[Epochs]${NC} - Total number of training epochs"
echo -e "  Default: ${YELLOW}$DEFAULT_EPOCHS${NC}"
echo -ne "  Enter epochs (press Enter for default): "
read -r EPOCHS
if [[ -z "$EPOCHS" ]]; then
    EPOCHS="$DEFAULT_EPOCHS"
fi
echo -e "  ${GREEN}✓${NC} Training for $EPOCHS epochs"
echo ""

echo -e "  ${GREEN}[Batch Size]${NC} - Number of samples per GPU per iteration"
echo -e "  Default: ${YELLOW}$DEFAULT_BATCH_SIZE${NC} (adjust based on GPU memory)"
echo -ne "  Enter batch size (press Enter for default): "
read -r BATCH_SIZE
if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
fi
echo -e "  ${GREEN}✓${NC} Batch size: $BATCH_SIZE per GPU"
echo ""

echo -e "  ${GREEN}[Gradient Accumulation]${NC} - Accumulate gradients over N steps"
echo -e "  Effective batch = batch_size × accumulation_steps"
echo -e "  Default: ${YELLOW}$DEFAULT_GRADIENT_ACCUMULATION${NC} (effective batch = $((DEFAULT_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION)))"
echo -ne "  Enter accumulation steps (press Enter for default): "
read -r GRADIENT_ACCUMULATION
if [[ -z "$GRADIENT_ACCUMULATION" ]]; then
    GRADIENT_ACCUMULATION="$DEFAULT_GRADIENT_ACCUMULATION"
fi
EFFECTIVE_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION))
echo -e "  ${GREEN}✓${NC} Gradient accumulation: $GRADIENT_ACCUMULATION (effective batch: $EFFECTIVE_BATCH)"

echo ""
echo -e "${BLUE}>> Device Configuration${NC}"
echo ""
echo -e "${YELLOW}Supported devices: CUDA GPU, Ascend NPU, CPU${NC}"
echo ""

echo -e "  ${GREEN}[Multi-GPU Mode]${NC} - Use distributed training across multiple GPUs"
echo -e "  Default: ${YELLOW}No (single device)${NC}"
echo -ne "  Enable multi-GPU? [y/N]: "
read -r USE_MULTI_GPU_INPUT
if [[ "$USE_MULTI_GPU_INPUT" =~ ^[Yy] ]]; then
    USE_MULTI_GPU="y"
else
    USE_MULTI_GPU="n"
fi

if [[ "$USE_MULTI_GPU" == "y" ]]; then
    # Multi-GPU mode
    echo -e "  ${GREEN}✓${NC} Multi-GPU mode enabled"
    echo ""

    echo -e "  ${GREEN}[Number of GPUs]${NC} - How many GPUs to use for training"
    echo -e "  Default: ${YELLOW}$DEFAULT_NUM_GPUS${NC}"
    echo -ne "  Enter number of GPUs (press Enter for default): "
    read -r NUM_GPUS
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS="$DEFAULT_NUM_GPUS"
    fi
    echo -e "  ${GREEN}✓${NC} Using $NUM_GPUS GPUs"
    echo ""

    echo -e "  ${GREEN}[Device Type]${NC} - Hardware accelerator type"
    echo -e "  Options: ${YELLOW}cuda${NC} (NVIDIA GPU), ${YELLOW}npu${NC} (Ascend NPU)"
    echo -e "  Default: ${YELLOW}cuda${NC}"
    echo -ne "  Enter device type (press Enter for default): "
    read -r DEVICE_TYPE
    if [[ -z "$DEVICE_TYPE" ]]; then
        DEVICE_TYPE="cuda"
    fi
    echo -e "  ${GREEN}✓${NC} Device type: $DEVICE_TYPE"

    # Ask for visible devices
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        echo ""
        echo -e "  ${GREEN}[CUDA Visible Devices]${NC} - Specify which GPUs to use"
        echo -e "  Example: ${YELLOW}0,1,2,3${NC} for first 4 GPUs, leave empty for all"
        echo -ne "  Enter GPU IDs (press Enter to use all): "
        read -r VISIBLE_DEVICES
        if [[ -n "$VISIBLE_DEVICES" ]]; then
            echo -e "  ${GREEN}✓${NC} Using GPUs: $VISIBLE_DEVICES"
        else
            echo -e "  ${GREEN}✓${NC} Using all available GPUs"
        fi
    else
        VISIBLE_DEVICES=""
    fi

    DISTRIBUTED="yes"
    DEVICE=""  # Will be auto-detected in distributed mode
else
    # Single-GPU mode
    echo -e "  ${GREEN}✓${NC} Single device mode"
    echo ""

    echo -e "  ${GREEN}[Device]${NC} - Which device to use for training"
    echo -e "  Options: ${YELLOW}cuda:0${NC}, ${YELLOW}cuda:1${NC}, ${YELLOW}npu:0${NC}, ${YELLOW}cpu${NC}"
    echo -e "  Default: ${YELLOW}$DEFAULT_DEVICE${NC}"
    echo -ne "  Enter device (press Enter for default): "
    read -r DEVICE
    if [[ -z "$DEVICE" ]]; then
        DEVICE="$DEFAULT_DEVICE"
    fi
    echo -e "  ${GREEN}✓${NC} Using device: $DEVICE"

    NUM_GPUS=1
    DISTRIBUTED="no"
    VISIBLE_DEVICES=""
fi

echo ""
echo -e "${BLUE}>> Resume Training${NC}"
echo ""

echo -ne "${GREEN}Resume from checkpoint? (leave empty for new training)${NC}: "
read -r RESUME

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}                    Training Summary${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  Train config:  ${YELLOW}$TRAIN_CONFIG${NC}"
echo -e "  Val config:    ${YELLOW}$VAL_CONFIG${NC}"
echo -e "  Epochs:        ${YELLOW}$EPOCHS${NC}"
echo -e "  Batch size:    ${YELLOW}$BATCH_SIZE${NC} (per GPU)"
echo -e "  Grad accum:    ${YELLOW}$GRADIENT_ACCUMULATION${NC} steps"
if [[ "$DISTRIBUTED" == "yes" ]]; then
    echo -e "  Multi-GPU:     ${YELLOW}Yes ($NUM_GPUS GPUs)${NC}"
    if [[ -n "$VISIBLE_DEVICES" ]]; then
        echo -e "  Visible GPUs:  ${YELLOW}$VISIBLE_DEVICES${NC}"
    fi
    TOTAL_EFFECTIVE=$((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))
    echo -e "  Effective batch: ${YELLOW}$TOTAL_EFFECTIVE${NC} ($BATCH_SIZE × $GRADIENT_ACCUMULATION × $NUM_GPUS)"
else
    echo -e "  Multi-GPU:     ${YELLOW}No (single device)${NC}"
    echo -e "  Device:        ${YELLOW}$DEVICE${NC}"
    echo -e "  Effective batch: ${YELLOW}$EFFECTIVE_BATCH${NC} ($BATCH_SIZE × $GRADIENT_ACCUMULATION)"
fi
if [[ -n "$RESUME" ]]; then
    echo -e "  Resume from:   ${YELLOW}$RESUME${NC}"
    echo -e "  Mode:          ${YELLOW}Resume training${NC}"
else
    echo -e "  Mode:          ${YELLOW}New training${NC}"
fi
# These parameters are used for both new and resume training
echo -e "  Adapters:      ${YELLOW}SwinIR + Restormer + X-Restormer${NC}"
echo -e "  Freeze cores:  ${YELLOW}None (all trainable)${NC}"
echo -e "  Backbone LR:   ${YELLOW}1e-6${NC} (low to prevent forgetting)"
echo -e "  Adapter LR:    ${YELLOW}3e-4${NC}"
echo -e "  Warmup:        ${YELLOW}1 epoch${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Confirmation
echo -ne "${GREEN}Start training? [Y/n]${NC}: "
read -r confirm
if [[ "$confirm" =~ ^[Nn] ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Activating conda environment: tool${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tool

# Set cache directory (for model weights: VGG, LPIPS, MUSIQ, CLIPIQA, etc.)
# This allows running on machines without root access to ~/.cache
CACHE_DIR="$CHAIN_ROOT/cache"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HOME="$CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$CACHE_DIR"
echo -e "${GREEN}✓${NC} Cache directory: $CACHE_DIR"

# Build training command
if [[ "$DISTRIBUTED" == "yes" ]]; then
    # Multi-GPU distributed training
    CMD_PREFIX=""
    if [[ -n "$VISIBLE_DEVICES" ]]; then
        CMD_PREFIX="CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES "
    fi
    CMD="${CMD_PREFIX}torchrun --nproc_per_node=$NUM_GPUS training/train_combined.py"
    CMD="$CMD --distributed"
else
    # Single-GPU training
    CMD="python3 training/train_combined.py"
    CMD="$CMD --device $DEVICE"
fi

CMD="$CMD --train-config $TRAIN_CONFIG"
CMD="$CMD --val-config $VAL_CONFIG"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --gradient-accumulation $GRADIENT_ACCUMULATION"
CMD="$CMD --grad-clip-norm 2"

# Add resume flag if specified
if [[ -n "$RESUME" ]]; then
    if [[ ! -d "$RESUME" ]]; then
        echo -e "${RED}Error: Resume directory not found: $RESUME${NC}"
        exit 1
    fi
    CMD="$CMD --resume $RESUME"

    # NOTE: For resume training, we still need to specify adapter and learning rate parameters
    # because they are not saved in the checkpoint (only model weights are saved)
    # Loss scheduling parameters will be restored from checkpoint, so we don't specify them

    # Adapter parameters for stable cascade training (all enabled, none frozen)
    CMD="$CMD --use-swinir-adapter --no-freeze-swinir"
    CMD="$CMD --use-restormer-adapter --no-freeze-restormer"
    CMD="$CMD --use-xrestormer-adapter --no-freeze-xrestormer"

    # Differential learning rate (prevent catastrophic forgetting)
    # - backbone_lr: 1e-6 (very low to protect pretrained weights)
    # - adapter_lr: 3e-4 (normal rate for adapter layers)
    CMD="$CMD --backbone-lr 1e-6"
    CMD="$CMD --adapter-lr 3e-4"
    CMD="$CMD --warmup-epochs 1"
else
    # Progressive loss scheduling parameters (only for new training)
    CMD="$CMD --transition-ratio 0.3"
    CMD="$CMD --target-pixel 0.4"
    CMD="$CMD --target-perceptual 0.10"
    CMD="$CMD --target-lpips 0.15"
    CMD="$CMD --target-musiq 0.10"
    CMD="$CMD --target-clipiqa 0.10"

    # Adapter parameters for stable cascade training (all enabled, none frozen)
    CMD="$CMD --use-swinir-adapter --no-freeze-swinir"
    CMD="$CMD --use-restormer-adapter --no-freeze-restormer"
    CMD="$CMD --use-xrestormer-adapter --no-freeze-xrestormer"

    # Differential learning rate (prevent catastrophic forgetting)
    # - backbone_lr: 1e-6 (very low to protect pretrained weights)
    # - adapter_lr: 3e-4 (normal rate for adapter layers)
    CMD="$CMD --backbone-lr 1e-6"
    CMD="$CMD --adapter-lr 3e-4"
    CMD="$CMD --warmup-epochs 1"
fi

echo ""
echo -e "${BLUE}Running:${NC} $CMD"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Run training
eval $CMD

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}                 Training completed!${NC}"
echo -e "${GREEN}============================================================${NC}"
