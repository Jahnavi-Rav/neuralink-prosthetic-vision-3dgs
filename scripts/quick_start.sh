#!/bin/bash
# Quick Start Script for Neuralink Prosthetic Vision 3DGS

echo "🚀 Setting up Neuralink Prosthetic Vision 3DGS Project..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n prosthetic-vision python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate prosthetic-vision

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download sample data (optional)
echo "📂 Downloading sample Replica scene..."
mkdir -p data/replica
# TODO: Add download script

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  conda activate prosthetic-vision"
echo "  python src/gaussian_splatting/train.py"
