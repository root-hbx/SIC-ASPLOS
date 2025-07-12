#!/bin/bash

# Install script for UNet Flood Segmentation
echo "Setting up UNet Flood Segmentation environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda to install PyTorch with MPS support..."
    conda install pytorch torchvision -c pytorch
else
    echo "Using pip to install PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other dependencies..."
pip3 install -r requirements.txt

echo "Installation completed!"
echo "You can now run: python unet.py"
