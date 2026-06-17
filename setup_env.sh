#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Environment Setup
# =============================================================================
# Run this script from the project root to set up the virtual environment.
# Usage: bash setup_env.sh
# =============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
SAM2_ROOT="${SAM2_ROOT:-$PROJECT_ROOT/../sam2}"

echo "=============================================="
echo "Pollen AI Atlas - Environment Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Using: $PYTHON_VERSION"

# Check for minimum Python 3.10
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Virtual environment already exists at: $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
else
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Created: $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install core requirements
echo ""
echo "Installing core requirements..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# =============================================================================
# SAM-2 Installation (Required for Mining)
# =============================================================================
echo ""
echo "=============================================="
echo "SAM-2 Setup"
echo "=============================================="

if [ -d "$SAM2_ROOT" ]; then
    echo "SAM-2 found at: $SAM2_ROOT"
    echo ""
    echo "Installing SAM-2 from local clone..."
    pip install -e "$SAM2_ROOT"
    echo "SAM-2 installed"
    
    # Check for checkpoints
    if [ -f "${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt" ]; then
        echo "SAM-2 checkpoint found: sam2.1_hiera_large.pt"
    else
        echo "SAM-2 checkpoint not found. Download from:"
        echo "  https://github.com/facebookresearch/sam2#model-checkpoints"
    fi
else
    echo "SAM-2 not found at: $SAM2_ROOT"
    echo ""
    echo "To install SAM-2:"
    echo "  git clone https://github.com/facebookresearch/sam2.git $SAM2_ROOT"
    echo "  pip install -e $SAM2_ROOT"
    echo ""
    echo "Or install from PyPI:"
    echo "  pip install git+https://github.com/facebookresearch/sam2.git"
fi

# =============================================================================
# vLLM Installation (Required for Captioning)
# =============================================================================
echo ""
echo "=============================================="
echo "vLLM Setup (for Captioning Phase)"
echo "=============================================="
echo "vLLM requires CUDA and is best installed separately:"
echo ""
echo "  pip install vllm"
echo ""
echo "Serve the captioning model with:"
echo "  vllm serve qwen25-vl-32b-awq \\"
echo "    --dtype float16 --quantization awq_marlin \\"
echo "    --task generate --limit-mm-per-prompt 'image=2' \\"
echo "    --max-model-len 8192 --gpu-memory-utilization 0.80 \\"
echo "    --port 11446 --tensor-parallel-size 2"
echo ""

# =============================================================================
# OpenSlide System Library
# =============================================================================
echo "=============================================="
echo "OpenSlide System Library"
echo "=============================================="

if command -v openslide-show-properties &> /dev/null; then
    echo "OpenSlide system library found"
else
    echo "OpenSlide system library not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install openslide-tools"
    echo "  macOS: brew install openslide"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c \"import torch, timm, h5py, openslide; import lib.model; print('OK')\""
echo ""
echo "Key paths configured in bash scripts:"
echo "  Project:     $PROJECT_ROOT"
echo "  SAM-2:       $SAM2_ROOT"
echo "  SAM-2 ckpt:  $SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt"
echo "  Data:        set DATA_ROOT to the external data root"
echo "=============================================="
