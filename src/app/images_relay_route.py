import base64
import json
import logging
import time
import uuid
import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.app.supabse_route import admin_required, token_required
from src.core.image_similarity import ImageSimilarity
from src.core.vector_index import VectorIndex
from src.db.db_connect import get_connection
from src.repo.split_images_repo import select_multiple_image_data_by_ids
from src.utils.data_conversion import base64_to_numpy
from src.utils.request_tracker import request_tracker, CancellationException

# Define a Blueprint to organize routes
api_rp = Blueprint('images_relay', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread-safe instance of ImageSimilarity
image_similarity = ImageSimilarity()
vector_index = VectorIndex()


@api_rp.route("/relay_image", methods=["POST"])
@admin_required
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

        # Image conversion - wrap in try-except to handle cancellation
        try:
            convert_start = time.time()

            # Make this operation cancellable
            def convert_image():
                image_np = base64_to_numpy(base64_image)
                print(f"Image conversion time: {time.time() - convert_start:.4f} seconds")
                return image_np

            image_np = request_tracker.run_cancellable(request_id, convert_image)

        except CancellationException:
            print(f"Request {request_id} was cancelled during image conversion")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        # Feature extraction - wrap in try-except to handle cancellation
        try:
            feature_start = time.time()

            # Make feature extraction cancellable
            def extract_features():
                return image_similarity.extract_feature(image_np)

            image_feature = request_tracker.run_cancellable(request_id, extract_features)
            print(f"Feature extraction time: {time.time() - feature_start:.4f} seconds")

        except CancellationException:
            print(f"Request {request_id} was cancelled during feature extraction")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        # Find similar images using vector index - wrap in try-except for cancellation
        try:
            search_start = time.time()

            # Make vector search cancellable
            def search_vectors():
                return vector_index.search_similar_images(image_feature, num)

            similarity_pairs = request_tracker.run_cancellable(request_id, search_vectors)
            print(f"Vector search time: {time.time() - search_start:.4f} seconds")

        except CancellationException:
            print(f"Request {request_id} was cancelled during vector search")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        if not similarity_pairs:
            print("No similar images found")
            request_tracker.complete_request(request_id)
            return jsonify([])

        # Fetch image data in batch - wrap in try-except for cancellation
        try:
            image_fetch_start = time.time()
            top_ids = [id for id, _ in similarity_pairs]

            # Make database fetch cancellable
            def fetch_images():
                return select_multiple_image_data_by_ids(top_ids)

            all_image_data = request_tracker.run_cancellable(request_id, fetch_images)

        except CancellationException:
            print(f"Request {request_id} was cancelled during database fetch")
            request_tracker.complete_request(request_id)
            return jsonify({"status": "cancelled", "message": "Processing cancelled by user"}), 200

        result = []
        # Process results - check for cancellation during iteration
        for idx, sim in similarity_pairs:
            # Check for cancellation directly in the loop
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


# Add a new endpoint to support frontend's recent requests queries
@api_rp.route("/request_status/recent", methods=["GET"])
@cross_origin()
def recent_requests():
    """
    Get recent active requests - useful for the frontend to find which requests to cancel
    """
    with request_tracker.request_lock:
        current_time = time.time()
        recent = [
            {
                "id": req_id,
                "timestamp": info["start_time"],
                "age": current_time - info["start_time"]
            }
            for req_id, info in request_tracker.active_requests.items()
        ]
        # Sort by timestamp (newest first)
        recent.sort(key=lambda x: x["timestamp"], reverse=True)

    return jsonify({"requests": recent}), 200


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


@api_rp.route("/cleanup_requests", methods=["POST"])
@cross_origin()
def cleanup_requests():
    """
    Administrative endpoint to clean up stale requests
    """
    max_age = request.json.get("max_age_seconds", 3600)  # Default: 1 hour
    request_tracker.cleanup_old_requests(max_age)
    return jsonify({"status": "success", "message": "Cleaned up stale requests"}), 200