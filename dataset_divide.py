import os
import shutil
import glob
from pathlib import Path

def move_files_by_range(source_dir, target_dir, start_index, end_index):
    """Move files within a specified range from source directory to target directory
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        start_index: Start index (1-based counting)
        end_index: End index (inclusive)
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files and sort them
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_files = []
    
    for ext in image_extensions:
        files = glob.glob(os.path.join(source_dir, ext))
        files.extend(glob.glob(os.path.join(source_dir, ext.upper())))
        all_files.extend(files)
    
    # Keep original file order (no sorting)
    
    print(f"Found {len(all_files)} image files in {source_dir}")
    
    # Check if index range is valid
    if start_index < 1 or end_index > len(all_files):
        print(f"Warning: Index range {start_index}-{end_index} exceeds file count {len(all_files)}")
        start_index = max(1, start_index)
        end_index = min(len(all_files), end_index)
        print(f"Adjusted to: {start_index}-{end_index}")
    
    # Convert to 0-based index
    start_idx = start_index - 1
    end_idx = end_index - 1
    
    # Move files in specified range
    moved_count = 0
    for i in range(start_idx, end_idx + 1):
        if i < len(all_files):
            source_file = all_files[i]
            filename = os.path.basename(source_file)
            target_file = os.path.join(target_dir, filename)
            
            try:
                shutil.move(source_file, target_file)
                moved_count += 1
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving file {filename}: {e}")
    
    print(f"Successfully moved {moved_count} files from {source_dir} to {target_dir}")
    
    return moved_count

def main():
    """Main function"""
    # Define paths
    base_dir = Path(__file__).parent
    train_dataset_dir = base_dir / "train_dataset"
    test_dataset_dir = base_dir / "test_dataset"
    
    # Source and target directories
    source_images_dir = train_dataset_dir / "Images"
    source_masks_dir = train_dataset_dir / "Masks"
    target_images_dir = test_dataset_dir / "Images"
    target_masks_dir = test_dataset_dir / "Masks"
    
    # Check if source directories exist
    if not source_images_dir.exists():
        print(f"Error: Source Images directory does not exist: {source_images_dir}")
        return
    
    if not source_masks_dir.exists():
        print(f"Error: Source Masks directory does not exist: {source_masks_dir}")
        return
    
    # Get file count dynamically from the source directories
    # Get Images count to determine the range
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_image_files = []
    
    for ext in image_extensions:
        files = glob.glob(os.path.join(str(source_images_dir), ext))
        files.extend(glob.glob(os.path.join(str(source_images_dir), ext.upper())))
        all_image_files.extend(files)
    
    total_files = len(all_image_files)
    
    # Define the range of files to move (first 5% of files as test set)
    test_ratio = 0.05  # 5% for test set
    start_index = 1  # Start from the beginning
    end_index = int(total_files * test_ratio)  # First 5%
    
    print(f"Moving first {test_ratio:.0%} of files (approximately {end_index} files)")
    print(f"File range: {start_index}-{end_index}")
    
    print(f"Starting to move files {start_index}-{end_index} from train_dataset to test_dataset")
    print("=" * 60)
    
    # Move Images
    print("Moving Images...")
    images_moved = move_files_by_range(
        str(source_images_dir), 
        str(target_images_dir), 
        start_index, 
        end_index
    )
    
    print("\n" + "=" * 60)
    
    # Move Masks
    print("Moving Masks...")
    masks_moved = move_files_by_range(
        str(source_masks_dir), 
        str(target_masks_dir), 
        start_index, 
        end_index
    )
    
    print("\n" + "=" * 60)
    print("Dataset division completed!")
    print(f"Total moved: {images_moved} image files and {masks_moved} mask files")
    
    # Verify results
    if target_images_dir.exists():
        images_count = len(list(target_images_dir.glob("*.*")))
        print(f"test_dataset/Images now contains {images_count} files")
    
    if target_masks_dir.exists():
        masks_count = len(list(target_masks_dir.glob("*.*")))
        print(f"test_dataset/Masks now contains {masks_count} files")

if __name__ == "__main__":
    main()


