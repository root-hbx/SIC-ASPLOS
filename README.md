# SIC UNet

A deep-learning project based on Space Intelligence and Edge Computing for flood area segmentation using UNet architecture, ready to go for SIC (ASPLOS'26).

## Quick Start

(1) **Install dependencies**

```bash
sudo chmod +x ./install.sh
conda activate [YOUR_CONDA_VENV]
./install.sh
```

(2) **Prepare your data**

- Place training images in `train_dataset/Images/`
- Place corresponding masks in `train_dataset/Masks/`
- Use `dataset_divide.py` if needed to split data

(3) **Train the model**

```bash
python unet.py
```

(4) **Use well-trained model to test**

```bash
python evaluate.py
```

## Key Features

- **M2 Pro Optimized**: Automatic MPS (Metal Performance Shaders) detection
- **Modular Design**: Clean separation of model classes and training logic  
- **Advanced Loss**: Combined BCE + Dice loss for better segmentation
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Model Persistence**: Automatic saving of best performing models

## Output Files

After training:

- `best_flood_unet_model.pth` - Best model weights
- `final_flood_unet_model.pth` - Final model weights  
- `training_history.png` - Training curves visualization

## Model Architecture

UNet with skip connections:

- **Input**: RGB images (256×256)
- **Output**: Binary flood masks
- **Encoder**: 5 downsampling levels (64→1024 channels)
- **Decoder**: 4 upsampling levels with skip connections
- **Parameters**: ~31M trainable parameters


