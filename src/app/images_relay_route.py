import base64
import json
import logging
import time
import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.core.image_similarity import ImageSimilarity
from src.core.vector_index import VectorIndex
from src.db.db_connect import get_connection
from src.repo.split_images_repo import select_multiple_image_data_by_ids
from src.utils.data_conversion import base64_to_numpy

# Define a Blueprint to organize routes
api_rp = Blueprint('images_relay', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize instances
similarity = ImageSimilarity()
vector_index = VectorIndex()

@api_rp.route("/relay_image", methods=["POST"])
@cross_origin()
def image_relay():
    start_time = time.time()

    try:
        data = request.get_json()
        num = data.get('num', 5)  # Default to 5 if not specified
        base64_image = data.get('image_base64')

        if not base64_image:
            print("Error: No 'image_base64' field in request data")
            return jsonify({"error": "'image_base64' field is required"}), 400

        print(f"Return count: {num}, received image_base64: {base64_image[:30]}...")

        # Image conversion
        convert_start = time.time()
        image_np = base64_to_numpy(base64_image)
        print(f"Image conversion time: {time.time() - convert_start:.4f} seconds")

        # Feature extraction
        feature_start = time.time()
        image_feature = similarity.extract_feature(image_np)
        print(f"Feature extraction time: {time.time() - feature_start:.4f} seconds")

        # Find similar images using vector index
        search_start = time.time()
        similarity_pairs = vector_index.search_similar_images(image_feature, num)
        print(f"Vector search time: {time.time() - search_start:.4f} seconds")

        if not similarity_pairs:
            print("No similar images found")
            return jsonify([])

        # Fetch image data in batch
        image_fetch_start = time.time()
        top_ids = [id for id, _ in similarity_pairs]

        # Get all image data for the IDs
        all_image_data = select_multiple_image_data_by_ids(top_ids)

        result = []
        for idx, sim in similarity_pairs:
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

        return jsonify(result)

    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error occurred! Total execution time: {total_time:.4f} seconds")
        print(f"Error details: {e}")
        return jsonify({"error": str(e), "execution_time": total_time}), 500