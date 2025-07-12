import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


from classes import UNet

# =====================
# --- Configuration ---
# =====================

MODEL_PATH = "best_flood_unet_model.pth"
TEST_IMG_DIR = "test_dataset/Images/"
TEST_MASK_DIR = "test_dataset/Masks/"
OUTPUT_DIR = "test_dataset/test_case_mask/"
IMAGE_HEIGHT = 256  # Must be the same dimensions as used during training
IMAGE_WIDTH = 256   # Must be the same dimensions as used during training


def calculate_metrics(pred_mask, true_mask):
    """Calculate IoU and Dice Score."""
    # Ensure inputs are tensors on the CPU
    pred_mask = pred_mask.cpu()
    true_mask = true_mask.cpu()

    # Convert prediction and ground truth masks to boolean
    pred_bool = (pred_mask > 0.5)
    true_bool = (true_mask > 0.5)

    intersection = (pred_bool & true_bool).sum().float()
    union = (pred_bool | true_bool).sum().float()
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    
    # Intersection over Union (IoU)
    iou = (intersection + epsilon) / (union + epsilon)
    
    # Dice Score
    dice = (2. * intersection + epsilon) / (pred_bool.sum() + true_bool.sum() + epsilon)
    
    return iou.item(), dice.item()


def main():
    # =========================
    # --- 1. Initialization ---
    # =========================
    print("🚀 Starting model evaluation...")
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # --- 2. Load Model ---
    # =========================
    print(f"Loading model from: {MODEL_PATH}")
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    # Load the entire checkpoint dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Then, load only the model's state_dict from that dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() # Set the model to evaluation mode
    
    # =====================================
    # --- 3. Define Image Preprocessing ---
    # =====================================
    # Transform for input images
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform for mask images - ensures consistency with training
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # ===================================================
    # --- 4. Iterate over the test set for evaluation ---
    # ===================================================
    test_image_files = sorted(os.listdir(TEST_IMG_DIR))
    all_ious = []
    all_dices = []
    
    print(f"\nEvaluating on {len(test_image_files)} test images...")
    for img_file in tqdm(test_image_files, desc="Evaluating"):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        true_mask_path = os.path.join(TEST_MASK_DIR, img_file)
        output_mask_path = os.path.join(OUTPUT_DIR, img_file)

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Load and preprocess the ground truth mask using the consistent transform
        true_mask_pil = Image.open(true_mask_path).convert("L")
        true_mask_tensor = mask_transform(true_mask_pil)

        # Model prediction
        with torch.no_grad():
            logits = model(input_tensor)
            pred_probs = torch.sigmoid(logits) # Convert output to probabilities (0-1)
        
        # Save the generated mask image
        pred_mask_np = pred_probs.squeeze().cpu().numpy()
        pred_mask_img = (pred_mask_np > 0.5).astype(np.uint8) * 255
        Image.fromarray(pred_mask_img).save(output_mask_path)
        
        # Calculate metrics
        iou, dice = calculate_metrics(pred_probs.squeeze(), true_mask_tensor.squeeze())
        all_ious.append(iou)
        all_dices.append(dice)

    # ================================
    # --- 5. Display Final Results ---
    # ================================
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)

    print("\n" + "="*50)
    print("✅ Evaluation Complete!")
    print(f"📊 Average Intersection over Union (IoU): {avg_iou:.4f}")
    print(f"🎯 Average Dice Score: {avg_dice:.4f}")
    print("="*50)
    print(f"📁 All predicted masks have been saved to: {OUTPUT_DIR}")

    # ==============================================================
    # -------- 6. Visualize a random sample for comparison ---------
    # ==============================================================
    print("\nVisualizing a random sample for comparison...")
    random_idx = np.random.randint(0, len(test_image_files))
    sample_img_file = test_image_files[random_idx]
    
    original_img = Image.open(os.path.join(TEST_IMG_DIR, sample_img_file))
    true_mask_img = Image.open(os.path.join(TEST_MASK_DIR, sample_img_file))
    generated_mask_img = Image.open(os.path.join(OUTPUT_DIR, sample_img_file))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(true_mask_img, cmap='gray')
    axes[1].set_title("Ground Truth Mask", fontsize=16)
    axes[1].axis('off')

    axes[2].imshow(generated_mask_img, cmap='gray')
    axes[2].set_title("Predicted Mask", fontsize=16)
    axes[2].axis('off')
    
    plt.suptitle(f"Sample Comparison: {sample_img_file}\nIoU: {all_ious[random_idx]:.4f} | Dice: {all_dices[random_idx]:.4f}", fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


