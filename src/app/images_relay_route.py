import base64
import json
import logging
import time
import uuid
import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.core.image_similarity import ImageSimilarity
from src.core.vector_index import VectorIndex
from src.db.db_connect import get_connection
from src.repo.split_images_repo import select_multiple_image_data_by_ids
from src.utils.data_conversion import base64_to_numpy
from src.utils.request_tracker import request_tracker

# Define a Blueprint to organize routes
api_rp = Blueprint('images_relay', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_similarity = ImageSimilarity()
vector_index = VectorIndex()


@api_rp.route("/relay_image", methods=["POST"])
@cross_origin()
def image_relay():
    start_time = time.time()

    # Generate a unique request ID
    request_id = str(uuid.uuid4())

    # Register this request in the tracker
    request_tracker.register_request(request_id)

    try:
        data = request.get_json()
        num = data.get('num', 5)  # Default to 5 if not specified
        base64_image = data.get('image_base64')

        if not base64_image:
            print("Error: No 'image_base64' field in request data")
            request_tracker.complete_request(request_id)
            return jsonify({"error": "'image_base64' field is required"}), 400

        print(f"Return count: {num}, received image_base64: {base64_image[:30]}...")

        # Image conversion
        convert_start = time.time()
        image_np = base64_to_numpy(base64_image)
        print(f"Image conversion time: {time.time() - convert_start:.4f} seconds")

        # Check if request was cancelled during conversion
        if request_tracker.is_cancelled(request_id):
            print(f"Request {request_id} was cancelled after image conversion")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        # Feature extraction
        feature_start = time.time()
        image_feature = image_similarity.extract_feature(image_np)
        print(f"Feature extraction time: {time.time() - feature_start:.4f} seconds")

        # Check if request was cancelled during feature extraction
        if request_tracker.is_cancelled(request_id):
            print(f"Request {request_id} was cancelled after feature extraction")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        # Find similar images using vector index
        search_start = time.time()
        similarity_pairs = vector_index.search_similar_images(image_feature, num)
        print(f"Vector search time: {time.time() - search_start:.4f} seconds")

        # Check if request was cancelled during search
        if request_tracker.is_cancelled(request_id):
            print(f"Request {request_id} was cancelled after vector search")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        if not similarity_pairs:
            print("No similar images found")
            request_tracker.complete_request(request_id)
            return jsonify([])

        # Fetch image data in batch
        image_fetch_start = time.time()
        top_ids = [id for id, _ in similarity_pairs]

        # Get all image data for the IDs
        all_image_data = select_multiple_image_data_by_ids(top_ids)

        # Check if request was cancelled after fetching image data
        if request_tracker.is_cancelled(request_id):
            print(f"Request {request_id} was cancelled after fetching image data")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        result = []
        for idx, sim in similarity_pairs:
            # Check for cancellation during iteration (for large result sets)
            if request_tracker.is_cancelled(request_id):
                print(f"Request {request_id} was cancelled during result preparation")
                request_tracker.complete_request(request_id)
                return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

            image_data = all_image_data.get(idx)
            if image_data and 'splitted_image_data' in image_data:
                binary_string = image_data['splitted_image_data']
                base64_string = base64.b64encode(binary_string).decode("utf-8")

                result.append({
                    "id": idx,
                    "similarity": float(sim),  # Convert numpy float to Python float
                    "processed_image_base64": base64_string
                })
            else:
                print(f"Image data not found for ID {idx}")
                result.append({
                    "id": idx,
                    "similarity": float(sim),
                    "processed_image_base64": None
                })

        print(f"Image data fetch time: {time.time() - image_fetch_start:.4f} seconds")
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.4f} seconds")

        # Mark request as complete
        request_tracker.complete_request(request_id)

        # Include the request ID in the response
        return jsonify({
            "request_id": request_id,
            "results": result
        })

    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error occurred! Total execution time: {total_time:.4f} seconds")
        print(f"Error details: {e}")
        request_tracker.complete_request(request_id)
        return jsonify({"error": str(e), "execution_time": total_time}), 500


@api_rp.route("/cancel_request/<request_id>", methods=["POST"])
@cross_origin()
def cancel_request(request_id):
    """
    Endpoint to cancel an ongoing image processing request
    """
    success = request_tracker.cancel_request(request_id)
    if success:
        return jsonify({"status": "success", "message": f"Request {request_id} marked for cancellation"}), 200
    else:
        return jsonify({"status": "error", "message": f"Request {request_id} not found or already completed"}), 404


# Optional: Add a route to get request status
@api_rp.route("/request_status/<request_id>", methods=["GET"])
@cross_origin()
def request_status(request_id):
    """
    Get the status of a specific request
    """
    with request_tracker.request_lock:
        if request_id in request_tracker.active_requests:
            status = "cancelled" if request_tracker.active_requests[request_id]["cancelled"] else "processing"
            return jsonify({"status": status}), 200
        else:
            return jsonify({"status": "not_found"}), 404


# Optional: Add a maintenance route to clean up old requests
@api_rp.route("/cleanup_requests", methods=["POST"])
@cross_origin()
def cleanup_requests():
    """
    Administrative endpoint to clean up stale requests
    """
    max_age = request.json.get("max_age_seconds", 3600)  # Default: 1 hour
    request_tracker.cleanup_old_requests(max_age)
    return jsonify({"status": "success", "message": "Cleaned up stale requests"}), 200