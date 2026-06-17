#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# START vLLM SERVERS FOR VLM CAPTIONING
# =============================================================================
#
# Supports multiple model configurations:
#   - qwen25-awq:    Qwen2.5-VL-32B-AWQ (4-bit, ~17GB per GPU)
#   - qwen3-fp8:     Qwen3-VL-32B-Instruct-FP8 (~32GB)
#   - qwen35-fp8:    Qwen3.5-27B-FP8 VL (~27GB)
#   - qwen36-fp8:    Qwen3.6-27B-FP8 VL (~27GB)
#   - gemma4-nvfp4:  Gemma-4-31B-IT-NVFP4 (~18GB per GPU)
#   - gemma4-bf16:   Gemma-4-31B-IT bf16 (~59GB, A100 cluster)
#
# Usage:
#   ./start_vllm_concurrent.sh [model] [mode]
#
#   model: qwen25-awq (default) | qwen3-fp8 | qwen35-fp8 | qwen36-fp8 | gemma4-nvfp4 | gemma4-bf16
#   mode:  cluster (default, 8×A100) | dual4090 (2×RTX4090)
#
# Examples:
#   ./start_vllm_concurrent.sh                      # qwen25-awq on cluster
#   ./start_vllm_concurrent.sh qwen3-fp8            # qwen3-fp8 on cluster
#   ./start_vllm_concurrent.sh qwen35-fp8           # qwen3.5-fp8 on cluster
#   ./start_vllm_concurrent.sh qwen36-fp8 dual4090  # qwen3.6-fp8 on dual 4090
#   ./start_vllm_concurrent.sh gemma4-nvfp4 dual4090 # gemma4-nvfp4 on dual 4090
#   ./start_vllm_concurrent.sh gemma4-bf16 cluster   # gemma4-bf16 on A100 cluster
#   ./start_vllm_concurrent.sh qwen3-fp8 dual4090   # qwen3-fp8 on dual 4090
#
# To stop all servers:
#   pkill -f "vllm serve"
#
# =============================================================================

set -e

MODEL="${1:-qwen25-awq}"
MODE="${2:-cluster}"

# ============================================================
# CPU THREADING CONFIGURATION
# ============================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "CPU threading limits set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_MODEL_ROOT="${VLLM_MODEL_ROOT:-/path/to/local/vlm_models}"

case "$MODEL" in
    qwen25-awq|qwen25)
        MODEL_PATH="${QWEN25_AWQ_PATH:-${VLLM_MODEL_ROOT}/qwen25-vl-32b-awq}"
        MODEL_NAME="Qwen2.5-VL-32B-AWQ"
        QUANTIZATION_FLAG="--quantization awq_marlin"
        DTYPE_FLAG="--dtype float16"
        # AWQ is smaller, can handle more sequences
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=24
        GPU_UTIL_CLUSTER=0.95
        GPU_UTIL_DUAL=0.90
        ;;
    qwen3-fp8|qwen3)
        MODEL_PATH="${QWEN3_FP8_PATH:-${VLLM_MODEL_ROOT}/Qwen3-VL-32B-Instruct-FP8}"
        MODEL_NAME="Qwen3-VL-32B-Instruct-FP8"
        QUANTIZATION_FLAG=""  # FP8 is natively supported, no flag needed
        DTYPE_FLAG=""         # Model defines its own dtype
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=16
        GPU_UTIL_CLUSTER=0.90
        GPU_UTIL_DUAL=0.90
        ;;
    qwen36-fp8|qwen36)
        MODEL_PATH="${QWEN36_FP8_PATH:-${VLLM_MODEL_ROOT}/Qwen3.6-27B-FP8}"
        MODEL_NAME="Qwen3.6-27B-FP8"
        QUANTIZATION_FLAG=""  # FP8 natively quantized
        DTYPE_FLAG=""         # Model defines its own dtype
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=16
        GPU_UTIL_CLUSTER=0.90
        GPU_UTIL_DUAL=0.90
        ;;
    qwen35-fp8|qwen35)
        MODEL_PATH="${QWEN35_FP8_PATH:-${VLLM_MODEL_ROOT}/Qwen3.5-27B-FP8}"
        MODEL_NAME="Qwen3.5-27B-FP8"
        QUANTIZATION_FLAG=""  # FP8 natively quantized
        DTYPE_FLAG=""         # Model defines its own dtype
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=16
        GPU_UTIL_CLUSTER=0.90
        GPU_UTIL_DUAL=0.90
        ;;
    gemma4-nvfp4|gemma4)
        MODEL_PATH="${GEMMA4_NVFP4_PATH:-${VLLM_MODEL_ROOT}/Gemma-4-31B-IT-NVFP4}"
        MODEL_NAME="Gemma-4-31B-IT-NVFP4"
        QUANTIZATION_FLAG=""  # NVFP4 via modelopt, no vLLM flag needed
        DTYPE_FLAG=""         # Model defines its own dtype
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=16
        GPU_UTIL_CLUSTER=0.90
        GPU_UTIL_DUAL=0.90
        ;;
    gemma4-bf16)
        MODEL_PATH="${GEMMA4_BF16_PATH:-${VLLM_MODEL_ROOT}/gemma-4-31B-it}"
        MODEL_NAME="Gemma-4-31B-IT"
        QUANTIZATION_FLAG=""  # No quantization, native bf16
        DTYPE_FLAG="--dtype bfloat16"
        MAX_SEQS_CLUSTER=48
        MAX_SEQS_DUAL=16
        GPU_UTIL_CLUSTER=0.90
        GPU_UTIL_DUAL=0.90
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: qwen25-awq, qwen3-fp8, qwen35-fp8, qwen36-fp8, gemma4-nvfp4, gemma4-bf16"
        exit 1
        ;;
esac

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/03_captioning/vllm_logs}"
mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "MODEL: $MODEL_NAME"
echo "PATH:  $MODEL_PATH"
echo "MODE:  $MODE"
echo "============================================================"
echo ""

# ============================================================
# CLUSTER MODE: 8× A100 (one server per GPU)
# ============================================================
if [ "$MODE" == "cluster" ]; then
    echo "Starting 8 servers on ports 11446-11453..."
    echo ""
    
    for i in {0..7}; do
        port=$((11446 + i))
        log_file="${LOG_DIR}/vllm_${MODEL}_gpu${i}_port${port}.log"
        
        echo "Starting GPU $i on port $port..."
        
        CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL_PATH" \
            $DTYPE_FLAG \
            $QUANTIZATION_FLAG \
            --max-model-len 8192 \
            --gpu-memory-utilization $GPU_UTIL_CLUSTER \
            --max-num-seqs $MAX_SEQS_CLUSTER \
            --enable-chunked-prefill \
            --port $port \
            --tensor-parallel-size 1 \
            > "$log_file" 2>&1 &
        
        echo "  PID: $!"
        echo "  Log: $log_file"
        
        sleep 3
    done
    
    echo ""
    echo "============================================================"
    echo "All servers starting... Wait ~2-3 minutes"
    echo "============================================================"
    echo ""
    echo "To check status:"
    echo "  for port in {11446..11453}; do"
    echo "    echo -n \"Port \$port: \""
    echo "    curl -s http://localhost:\$port/v1/models | python3 -c \"import sys,json; print(json.load(sys.stdin)['data'][0]['id'])\" 2>/dev/null || echo \"NOT READY\""
    echo "  done"

# ============================================================
# DUAL 4090 MODE: Single server with TP=2
# ============================================================
elif [ "$MODE" == "dual4090" ]; then
    PORT=11446
    log_file="${LOG_DIR}/vllm_${MODEL}_dual4090.log"
    
    echo "Starting single server with TP=2 on port $PORT..."
    
    nohup vllm serve "$MODEL_PATH" \
        $DTYPE_FLAG \
        $QUANTIZATION_FLAG \
        --max-model-len 8192 \
        --gpu-memory-utilization $GPU_UTIL_DUAL \
        --max-num-seqs $MAX_SEQS_DUAL \
        --enable-chunked-prefill \
        --port $PORT \
        --tensor-parallel-size 2 \
        > "$log_file" 2>&1 &
    
    echo "  PID: $!"
    echo "  Log: $log_file"
    echo ""
    echo "Server starting... Wait ~1-2 minutes"
    echo ""
    echo "To check status:"
    echo "  curl -s http://localhost:$PORT/v1/models | jq"

else
    echo "Unknown mode: $MODE"
    echo "Available: cluster, dual4090"
    exit 1
fi

echo ""
echo "To stop all:"
echo "  pkill -f 'vllm serve'"
echo ""
echo "Logs in: $LOG_DIR"
