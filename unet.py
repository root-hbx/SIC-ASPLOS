import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from sklearn.metrics import jaccard_score
import warnings

# Import custom classes
from classes import FloodDataset, UNet, CombinedLoss

warnings.filterwarnings('ignore')

# Auto-detect best compute device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    return jaccard_score(target_flat, pred_flat, average='binary', zero_division=1)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train the UNet model"""
    
    train_losses = []
    val_losses = []
    val_ious = []
    best_val_iou = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # BCE + Dice Loss
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                iou = calculate_iou(outputs, masks)
                
                val_loss += loss.item()
                val_iou += iou
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_iou = val_iou / val_batches
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val IoU: {avg_val_iou:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_loss': avg_val_loss
            }, 'best_flood_unet_model.pth')
            print(f'New best model saved with IoU: {best_val_iou:.4f}')
    
    return train_losses, val_losses, val_ious


def plot_training_history(train_losses, val_losses, val_ious):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot IoU
    axes[1].plot(val_ious, label='Val IoU', color='green')
    axes[1].set_title('Validation IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot learning rate would go here if we tracked it
    axes[2].plot(train_losses, label='Train Loss')
    axes[2].set_title('Training Loss (Log Scale)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function"""
    
    # Hyperparameters
    BATCH_SIZE = 8  # Reduced for M2 Pro memory
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    IMAGE_SIZE = 256  # Reduced from 512 for faster training
    
    # Data paths
    train_images_dir = "train_dataset/Images"
    train_masks_dir = "train_dataset/Masks"
    test_images_dir = "test_dataset/Images"
    test_masks_dir = "test_dataset/Masks"
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FloodDataset(train_images_dir, train_masks_dir, train_transform, mask_transform)
    val_dataset = FloodDataset(test_images_dir, test_masks_dir, val_transform, mask_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Train the model
    train_losses, val_losses, val_ious = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_ious)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'image_size': IMAGE_SIZE
        }
    }, 'final_flood_unet_model.pth')
    
    print("Training completed!")
    print(f"Best validation IoU: {max(val_ious):.4f}")

if __name__ == "__main__":
    main()


