# src/app/image_processing_route.py
import logging
import time
import uuid
import base64
from pathlib import Path
import numpy as np
from PIL import Image
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin, CORS

from src.app.supabse_route import admin_required
from src.core.groundingdino_handler import ClothingDetector
from src.db.uploads.image_upload import ImageUploader
from src.utils.data_conversion import base64_to_numpy, numpy_to_base64
from src.utils.request_tracker import request_tracker, CancellationException
from src.core.vector_index import VectorIndex

# Define a Blueprint to organize routes
api_proc = Blueprint('image_processing', __name__)

CORS(api_proc)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

# Initialize required components
clothing_detector = ClothingDetector()
clothing_detector.box_threshold = 0.15  # Lower threshold for better sensitivity
image_uploader = ImageUploader()


def ensure_rgb_format(image_np):
    """Ensure image is in RGB format (3 channels)"""
    # Check the number of channels in the image
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        # If RGBA (4 channels), convert to RGB (3 channels)
        logger.info("Converting RGBA image to RGB format")
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Already RGB format
        return image_np
    elif len(image_np.shape) == 2:
        # If grayscale image (2D), convert to RGB
        logger.info("Converting grayscale image to RGB format")
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)
    else:
        # Handle other uncommon formats
        logger.warning(f"Unsupported image format: shape={image_np.shape}")
        # Try direct conversion
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)


@api_proc.route("/split_image", methods=["POST"])
@cross_origin()
def split_image():
    """
    Endpoint to split an image and return base64 encoded segments without storing in database
    
    Request format:
    {
        "image_base64": "base64 encoded image string",
        "force_process": true/false (optional, whether to process even if no clothing is detected)
    }
    
    Response format:
    {
        "success": true/false,
        "message": "status message",
        "request_id": "request ID, can be used for cancelling requests",
        "data": {
            "segments": [
                {
                    "segment_id": "segment_0",
                    "segment_base64": "base64 encoded segment image"
                },
                ...
            ]
        }
    }
    """
    start_time = time.time()

    # Generate unique request ID
    request_id = str(uuid.uuid4())

    # Register request in tracker
    request_tracker.register_request(request_id)

    try:
        # Get request data
        data = request.get_json()
        base64_image = data.get('image_base64')
        force_process = data.get('force_process', True)  # Default to force process

        # Parameter validation
        if not base64_image:
            logger.error("Missing 'image_base64' field in request data")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "Missing required parameter: 'image_base64'"
            }), 400

        logger.info(f"Starting image processing for request: {request_id}")

        # Step 1: Convert base64 to image array
        try:
            convert_start = time.time()

            # Define cancellable conversion function
            def convert_image():
                # Remove data prefix if present (e.g., "data:image/jpeg;base64,")
                if base64_image.startswith('data:'):
                    clean_base64 = base64_image.split(',', 1)[1]
                else:
                    clean_base64 = base64_image

                # Convert to numpy array
                image_np = base64_to_numpy(clean_base64)

                # Ensure image is in RGB format
                image_np = ensure_rgb_format(image_np)

                logger.info(f"Image conversion time: {time.time() - convert_start:.4f} seconds")
                logger.info(f"Image shape: {image_np.shape}")
                return image_np

            # Execute cancellable operation
            image_np = request_tracker.run_cancellable(request_id, convert_image)

        except CancellationException:
            logger.info(f"Request {request_id} cancelled during image conversion")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "Processing was cancelled by user",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"Image conversion error: {str(e)}",
                "request_id": request_id
            }), 400

        # Step 2: Use GroundingDINO to detect clothing items
        try:
            detect_start = time.time()

            # Define cancellable detection function
            def detect_clothes():
                return clothing_detector.detect_clothes(image_np)

            # Execute cancellable operation
            segmented_images = request_tracker.run_cancellable(request_id, detect_clothes)
            logger.info(f"Clothing detection time: {time.time() - detect_start:.4f} seconds")

            # Check if any clothing was detected
            if not segmented_images and not force_process:
                request_tracker.complete_request(request_id)
                return jsonify({
                    "success": False,
                    "message": "No clothing items detected in the image",
                    "request_id": request_id
                }), 200

            # If no clothing detected but force process is true, use the entire image
            if not segmented_images and force_process:
                logger.warning("No clothing detected, but processing entire image due to force_process flag")
                segmented_images = [image_np.copy()]

        except CancellationException:
            logger.info(f"Request {request_id} cancelled during clothing detection")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "Processing was cancelled by user",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"Clothing detection error: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"Clothing detection error: {str(e)}",
                "request_id": request_id
            }), 500

        # Step 3: Convert segmented images to base64
        segment_data = []
        for idx, img_array in enumerate(segmented_images):
            # Check if request has been cancelled
            if request_tracker.is_cancelled(request_id):
                logger.info(f"Request {request_id} cancelled during segment processing")
                request_tracker.complete_request(request_id)
                return jsonify({
                    "success": False,
                    "message": "Processing was cancelled by user",
                    "request_id": request_id
                }), 200

            # Ensure each segmented image is in RGB format
            img_array = ensure_rgb_format(img_array)
            
            # Convert numpy array to base64
            segment_base64 = numpy_to_base64(img_array, 'png')
            
            segment_data.append({
                "segment_id": f"segment_{idx}",
                "segment_base64": segment_base64
            })

        # Complete the request
        request_tracker.complete_request(request_id)
        
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.4f} seconds")
        
        return jsonify({
            "success": True,
            "message": f"Successfully split image into {len(segment_data)} segments",
            "request_id": request_id,
            "data": {
                "segments": segment_data
            }
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        request_tracker.complete_request(request_id)
        return jsonify({
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "request_id": request_id
        }), 500


@api_proc.route("/upload_single_image", methods=["POST"])
@cross_origin()
def upload_single_image():
    """
    Endpoint to upload a single image to the database
    
    Request format:
    {
        "image_base64": "base64 encoded image string",
        "original_image_id": "ID of the original image",
        "segment_id": "ID for this segment",
        "bounding_box": "bounding box information (optional)"
    }
    
    Response format:
    {
        "success": true/false,
        "message": "status message",
        "data": {
            "splitted_image_id": "ID of the uploaded image",
            "splitted_image_path": "path where the image is stored"
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()
        base64_img = data.get('image_base64')
        original_image_id = data.get('original_image_id')
        segment_id = data.get('segment_id')
        bounding_box = data.get('bounding_box', str(uuid.uuid4()))  # Default to random ID if not provided

        # Parameter validation
        if not base64_img or not original_image_id or not segment_id:
            missing_params = []
            if not base64_img: missing_params.append('image_base64')
            if not original_image_id: missing_params.append('original_image_id')
            if not segment_id: missing_params.append('segment_id')
            
            logger.error(f"Missing required parameters: {', '.join(missing_params)}")
            return jsonify({
                "success": False,
                "message": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400

        logger.info(f"Starting single image upload for segment: {segment_id}")

        # Step 1: Convert base64 to numpy array
        try:
            # Remove data prefix if present
            if base64_img.startswith('data:'):
                clean_base64 = base64_img.split(',', 1)[1]
            else:
                clean_base64 = base64_img

            # Convert to numpy array
            image_np = base64_to_numpy(clean_base64)
            
            # Ensure image is in RGB format
            image_np = ensure_rgb_format(image_np)
            
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return jsonify({
                "success": False,
                "message": f"Image conversion error: {str(e)}"
            }), 400

        # Step 2: Create necessary directories
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for the original image if it doesn't exist
        image_dir = data_dir / original_image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Save image to file system and database
        try:
            # Define file paths
            filename = f"{segment_id}.png"
            save_path = image_dir / filename
            relative_path = f"{original_image_id}/{filename}"
            
            # Create a unique ID for the split image
            unique_splitted_image_id = f"{original_image_id}_{segment_id}"

            # Extract features and save image
            vector = image_uploader.similarity_model.extract_feature(image_np)

            # Save image to file system
            img = Image.fromarray(image_np)
            img.save(save_path)

            # Upload to database
            image_uploader.upload_splitted_image_to_db(
                image_data=image_np,
                splitted_image_id=unique_splitted_image_id,
                splitted_image_path=relative_path,
                original_image_id=original_image_id,
                bounding_box=bounding_box,
                image_format="png",
                vector=vector
            )
            
            logger.info(f"Successfully uploaded image: {unique_splitted_image_id}")
            
            # 在上传成功后异步重建索引
            # 这里使用异步方式，不阻塞当前请求
            VectorIndex.async_rebuild_index()
            logger.info("Triggered asynchronous index rebuild after upload")
            
            return jsonify({
                "success": True,
                "message": "Image successfully uploaded",
                "data": {
                    "splitted_image_id": unique_splitted_image_id,
                    "splitted_image_path": relative_path
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error during image upload: {e}")
            return jsonify({
                "success": False,
                "message": f"Error during image upload: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }), 500


@api_proc.route("/cancel_processing/<request_id>", methods=["POST"])
@cross_origin()
def cancel_processing(request_id):
    """
    Cancel an ongoing image processing request
    
    Parameters:
        request_id: ID of the request to cancel
        
    Response format:
    {
        "success": true/false,
        "message": "status message"
    }
    """
    if request_tracker.cancel_request(request_id):
        logger.info(f"Request {request_id} cancelled by user")
        return jsonify({
            "success": True,
            "message": f"Processing request {request_id} cancelled"
        }), 200
    else:
        logger.warning(f"Could not cancel request {request_id}, not found or already completed")
        return jsonify({
            "success": False,
            "message": f"Request {request_id} not found or already completed"
        }), 404 