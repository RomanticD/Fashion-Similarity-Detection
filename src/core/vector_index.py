import numpy as np
import json
import logging
import os
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from src.repo.split_images_repo import select_all_vectors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define index file paths
ROOT_DIR = Path(__file__).parent.parent.parent
INDEX_FILE = ROOT_DIR / "vector_nn_index.pkl"
ID_MAP_FILE = ROOT_DIR / "vector_id_map.json"


def build_vector_index():
    """
    Build a Nearest Neighbors index from vectors stored in the database
    """
    logger.info("Building vector index from database vectors...")

    # Fetch all vectors from the database
    rows = select_all_vectors()
    if not rows:
        logger.error("No vectors found in the database")
        return None, None

    logger.info(f"Retrieved {len(rows)} vectors from database")

    # Extract vectors and IDs
    vectors = []
    ids = []

    for row in rows:
        try:
            vector_list = json.loads(row['vector'])
            vectors.append(vector_list)
            ids.append(int(row['id']))
        except Exception as e:
            logger.error(f"Error processing record {row['id']}: {e}")

    if not vectors:
        logger.error("No valid vectors could be processed")
        return None, None

    # Convert to numpy arrays
    vector_array = np.array(vectors, dtype=np.float32)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vector_array, axis=1, keepdims=True)
    # Handle zero norms
    norms[norms == 0] = 1.0
    normalized_vectors = vector_array / norms

    # Create index for cosine similarity
    # Use 'cosine' as the metric for directly computing cosine similarity
    index = NearestNeighbors(n_neighbors=min(20, len(vectors)),
                             algorithm='auto',
                             metric='cosine')
    index.fit(normalized_vectors)

    logger.info(f"Added {len(vectors)} vectors to Nearest Neighbors index")

    # Save ID mapping
    with open(ID_MAP_FILE, 'w') as f:
        json.dump(ids, f)
    logger.info(f"Saved ID mapping to {ID_MAP_FILE}")

    # Save index using pickle
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({
            'index': index,
            'vectors': normalized_vectors
        }, f)
    logger.info(f"Saved vector index to {INDEX_FILE}")

    return index, ids, normalized_vectors


def load_vector_index():
    """
    Load Nearest Neighbors index and ID mapping from files
    """
    if not INDEX_FILE.exists() or not ID_MAP_FILE.exists():
        logger.info("Index files not found, building new index")
        return build_vector_index()

    try:
        # Load index
        with open(INDEX_FILE, 'rb') as f:
            data = pickle.load(f)
            index = data['index']
            vectors = data['vectors']
        logger.info(f"Loaded vector index from {INDEX_FILE}")

        # Load ID mapping
        with open(ID_MAP_FILE, 'r') as f:
            ids = json.load(f)
        logger.info(f"Loaded ID mapping from {ID_MAP_FILE}")

        return index, ids, vectors
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        logger.info("Building new index")
        return build_vector_index()


def search_similar_images(feature_vector, num=5):
    """
    Search for similar images using Nearest Neighbors index

    Args:
        feature_vector: Feature vector of the query image
        num: Number of results to return

    Returns:
        List of (id, similarity) tuples
    """
    # Load index and ID mapping
    index, ids, vectors = load_vector_index()
    if index is None or ids is None:
        logger.error("Failed to load or build index")
        return []

    # Convert feature vector to numpy array and ensure correct shape
    query_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    # Normalize query vector for cosine similarity
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        query_norm = 1.0
    query_vector = query_vector / query_norm

    # Search the index
    k = min(num, len(ids))  # Can't retrieve more than what's in the index
    if k == 0:
        return []

    # Get distances and indices
    distances, indices = index.kneighbors(query_vector, n_neighbors=k)

    # Convert distances to similarities (sklearn returns cosine distance, which is 1-cosine_similarity)
    # So similarity = 1 - distance
    similarities = [1 - dist for dist in distances[0]]

    # Map indices back to database IDs
    results = [(ids[idx], sim) for idx, sim in zip(indices[0], similarities)]

    return results


def rebuild_index():
    """
    Force rebuild the index regardless of whether it exists
    """
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if ID_MAP_FILE.exists():
        ID_MAP_FILE.unlink()

    return build_vector_index()


if __name__ == "__main__":
    # Build index when run directly
    build_vector_index()