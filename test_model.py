import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

# Import custom classes
from classes import UNet

#TODO(bxhu): currently auto-gen by Claude. still W.I.P

# Auto-detect device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_trained_model(model_path):
    """Load the trained UNet model"""
    
    # Initialize model architecture
    model = UNet(n_channels=3, n_classes=1)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Best validation IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
    
    return model


def preprocess_image(image_path, image_size=256):
    """Preprocess input image for model prediction"""
    
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Apply same transforms as validation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 256, 256]
    
    return image_tensor, image, original_size


def predict_flood_mask(model, image_tensor):
    """Predict flood mask using trained model"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass through model
        logits = model(image_tensor)  # [1, 1, 256, 256]
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)  # [1, 1, 256, 256]
        
        # Convert to binary mask (threshold = 0.5)
        binary_mask = (probabilities > 0.5).float()
        
        # Remove batch and channel dimensions
        prob_mask = probabilities.squeeze().cpu().numpy()  # [256, 256]
        binary_mask = binary_mask.squeeze().cpu().numpy()  # [256, 256]
        
    return prob_mask, binary_mask


def visualize_results(original_image, prob_mask, binary_mask, save_path=None):
    """Visualize original image, probability mask, and binary prediction"""
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Probability heatmap
    im1 = axes[1].imshow(prob_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Flood Probability\n(0=No Flood, 1=Flood)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Binary prediction
    axes[2].imshow(binary_mask, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Binary Prediction\n(Blue=Flood Area)', fontsize=14)
    axes[2].axis('off')
    
    # Overlay on original
    overlay = np.array(original_image.resize((256, 256)))
    flood_overlay = np.zeros_like(overlay)
    flood_overlay[:, :, 2] = binary_mask * 255  # Blue channel for flood
    
    # Blend original with flood overlay
    blended = 0.7 * overlay + 0.3 * flood_overlay
    axes[3].imshow(blended.astype(np.uint8))
    axes[3].set_title('Flood Overlay\n(Blue=Detected Flood)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()


def calculate_flood_statistics(prob_mask, binary_mask):
    """Calculate statistics about detected flood area"""
    
    total_pixels = prob_mask.size
    flood_pixels = np.sum(binary_mask)
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    avg_flood_confidence = np.mean(prob_mask[binary_mask == 1]) if flood_pixels > 0 else 0
    
    print("\n" + "="*50)
    print("🌊 FLOOD DETECTION RESULTS")
    print("="*50)
    print(f"📊 Total image area: {total_pixels:,} pixels")
    print(f"🌊 Detected flood area: {flood_pixels:,} pixels")
    print(f"📈 Flood coverage: {flood_percentage:.2f}% of image")
    print(f"🎯 Average confidence: {avg_flood_confidence:.3f}")
    
    if flood_percentage > 50:
        print("⚠️  HIGH FLOOD RISK: Major flooding detected!")
    elif flood_percentage > 20:
        print("⚠️  MODERATE FLOOD RISK: Significant flooding detected!")
    elif flood_percentage > 5:
        print("⚠️  LOW FLOOD RISK: Minor flooding detected!")
    else:
        print("✅ MINIMAL FLOOD RISK: Little to no flooding detected!")
    
    print("="*50)


def test_single_image(model_path, image_path, save_results=True):
    """Test the trained model on a single image"""
    
    print(f"🔍 Testing model on: {image_path}")
    
    # Load model
    model = load_trained_model(model_path)
    
    # Preprocess image
    image_tensor, original_image, original_size = preprocess_image(image_path)
    print(f"📷 Original image size: {original_size}")
    
    # Predict
    prob_mask, binary_mask = predict_flood_mask(model, image_tensor)
    
    # Calculate statistics
    calculate_flood_statistics(prob_mask, binary_mask)
    
    # Visualize results
    if save_results:
        save_path = f"flood_detection_result_{os.path.basename(image_path)}.png"
    else:
        save_path = None
        
    visualize_results(original_image, prob_mask, binary_mask, save_path)
    
    return prob_mask, binary_mask


def test_multiple_images(model_path, test_folder="test_dataset/Images", num_samples=5):
    """Test the model on multiple images from test dataset"""
    
    print(f"🔍 Testing model on {num_samples} random images from {test_folder}")
    
    # Get list of test images
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"❌ No images found in {test_folder}")
        return
    
    # Load model once
    model = load_trained_model(model_path)
    
    # Randomly sample images
    import random
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    results = []
    for i, img_file in enumerate(sample_images):
        print(f"\n{'='*60}")
        print(f"🖼️  Testing Image {i+1}/{len(sample_images)}: {img_file}")
        print('='*60)
        
        image_path = os.path.join(test_folder, img_file)
        
        try:
            # Preprocess and predict
            image_tensor, original_image, _ = preprocess_image(image_path)
            prob_mask, binary_mask = predict_flood_mask(model, image_tensor)
            
            # Calculate statistics
            calculate_flood_statistics(prob_mask, binary_mask)
            
            # Store results
            flood_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
            results.append({
                'image': img_file,
                'flood_percentage': flood_percentage,
                'avg_confidence': np.mean(prob_mask[binary_mask == 1]) if np.sum(binary_mask) > 0 else 0
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    # Summary statistics
    if results:
        print(f"\n{'='*60}")
        print("📊 OVERALL SUMMARY")
        print('='*60)
        avg_flood = np.mean([r['flood_percentage'] for r in results])
        print(f"📈 Average flood coverage: {avg_flood:.2f}%")
        print(f"🎯 Images with >20% flooding: {sum(1 for r in results if r['flood_percentage'] > 20)}/{len(results)}")
        print(f"🎯 Images with >50% flooding: {sum(1 for r in results if r['flood_percentage'] > 50)}/{len(results)}")


def main():
    """Main function to demonstrate model usage"""
    
    print("🌊 FLOOD DETECTION MODEL DEMONSTRATION")
    print("="*60)
    
    # Check if trained model exists
    model_path = "best_flood_unet_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please train the model first by running: python unet.py")
        return
    
    # Test on multiple images from test dataset
    print("Testing on sample images from test dataset...")
    test_multiple_images(model_path, num_samples=3)
    
    print(f"\n{'='*60}")
    print("🎉 MODEL DEMONSTRATION COMPLETE!")
    print("="*60)
    print("💡 To test on your own image:")
    print("   test_single_image('best_flood_unet_model.pth', 'your_image.jpg')")

if __name__ == "__main__":
    main()
