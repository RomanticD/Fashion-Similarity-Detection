import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Set up Python path
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# Import related modules
from src.core.groundingdino_handler import ClothingDetector
from src.db.uploads.image_upload import ImageUploader


def process_single_image(image_path):
    """Process a single image and upload segmented parts"""
    print(f"Processing image: {image_path}")

    # Get image name (without extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Check if image file exists
    if not image_path.exists():
        print(f"Image file '{image_path}' does not exist, skipping.")
        return []

    try:
        # Open image and convert to numpy array
        image = Image.open(image_path)
        image_np = np.array(image)

        # Initialize detector and detect clothes
        detector = ClothingDetector()
        result_bboxes = detector.detect_clothes(image_np)

        # Create data directory
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for original image
        image_dir = data_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # Initialize uploader
        uploader = ImageUploader()

        # Process and upload each segmented image
        uploaded_paths = []
        for idx, img_array in enumerate(result_bboxes):
            filename = f"segment_{idx}.png"
            save_path = image_dir / filename

            # Process and upload image
            processed_path = uploader.process_and_upload_image(img_array, idx, save_path, image_name)
            if processed_path:
                uploaded_paths.append(processed_path)
                print(f"Uploaded segmented image: {processed_path}")

        return uploaded_paths

    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return []


def batch_process_screenshots():
    """Batch process all images in the ScreenShots directory"""
    # Get root directory
    root_dir = Path(__file__).parent.resolve()
    while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
        root_dir = root_dir.parent

    # Check if ScreenShots folder exists
    screenshots_dir = root_dir / "ScreenShots"
    if not screenshots_dir.exists() or not screenshots_dir.is_dir():
        raise FileNotFoundError(f"'{screenshots_dir}' folder does not exist. Please download resource files.")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(screenshots_dir.glob(f'*{ext}')))
        image_files.extend(list(screenshots_dir.glob(f'*{ext.upper()}')))  # Check uppercase extensions too

    if not image_files:
        print(f"No image files found in '{screenshots_dir}' directory.")
        return

    print(f"Found {len(image_files)} image files to process.")

    # Process all images
    all_uploaded_paths = []
    for image_path in image_files:
        uploaded_paths = process_single_image(image_path)
        all_uploaded_paths.extend(uploaded_paths)

    print(f"\nBatch processing complete! Uploaded {len(all_uploaded_paths)} segmented images.")
    return all_uploaded_paths


if __name__ == "__main__":
    try:
        # Execute batch processing
        uploaded_files = batch_process_screenshots()

        # Print processing results
        if uploaded_files:
            print("\nSuccessfully uploaded files:")
            for path in uploaded_files:
                print(f" - {path}")
        else:
            print("\nNo files were successfully uploaded.")

    except Exception as e:
        print(f"Error during batch processing: {e}")