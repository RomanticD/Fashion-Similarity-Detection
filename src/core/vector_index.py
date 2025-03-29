# src/core/vector_index.py
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


class VectorIndex:
    """
    A class for managing vector indices for fast similarity search.
    """

    def __init__(self, index_file=None, id_map_file=None):
        """
        Initialize the vector index manager.

        Args:
            index_file (Path, optional): Path to the index file.
            id_map_file (Path, optional): Path to the ID mapping file.
        """
        self.index_file = index_file or INDEX_FILE
        self.id_map_file = id_map_file or ID_MAP_FILE
        self.index = None
        self.ids = None
        self.vectors = None

    def build_index(self):
        """
        Build a nearest neighbors index from vectors in the database.

        Returns:
            tuple: (index, ids, vectors) or (None, None, None) if no vectors found.
        """
        logger.info("Building vector index from database vectors...")

        # Fetch all vectors from the database
        rows = select_all_vectors()
        if not rows:
            logger.error("No vectors found in the database")
            return None, None, None

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
            return None, None, None

        # Convert to numpy arrays
        vector_array = np.array(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vector_array, axis=1, keepdims=True)
        # Handle zero norms
        norms[norms == 0] = 1.0
        normalized_vectors = vector_array / norms

        # Create index for cosine similarity
        index = NearestNeighbors(
            n_neighbors=min(20, len(vectors)),
            algorithm='auto',
            metric='cosine'
        )
        index.fit(normalized_vectors)

        logger.info(f"Added {len(vectors)} vectors to Nearest Neighbors index")

        # Save ID mapping
        with open(self.id_map_file, 'w') as f:
            json.dump(ids, f)
        logger.info(f"Saved ID mapping to {self.id_map_file}")

        # Save index using pickle
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'index': index,
                'vectors': normalized_vectors
            }, f)
        logger.info(f"Saved vector index to {self.index_file}")

        self.index = index
        self.ids = ids
        self.vectors = normalized_vectors

        return index, ids, normalized_vectors

    def load_index(self):
        """
        Load the nearest neighbors index and ID mapping from files.

        Returns:
            tuple: (index, ids, vectors) or build a new index if files don't exist.
        """
        if not self.index_file.exists() or not self.id_map_file.exists():
            logger.info("Index files not found, building new index")
            return self.build_index()

        try:
            # Load index
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
                index = data['index']
                vectors = data['vectors']
            logger.info(f"Loaded vector index from {self.index_file}")

            # Load ID mapping
            with open(self.id_map_file, 'r') as f:
                ids = json.load(f)
            logger.info(f"Loaded ID mapping from {self.id_map_file}")

            self.index = index
            self.ids = ids
            self.vectors = vectors

            return index, ids, vectors
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Building new index")
            return self.build_index()

    def search_similar_images(self, feature_vector, num=5):
        """
        Search for similar images using the vector index.

        Args:
            feature_vector: The feature vector of the query image.
            num (int): The number of results to return.

        Returns:
            list: A list of (id, similarity) tuples.
        """
        # Load index if needed
        if self.index is None or self.ids is None or self.vectors is None:
            self.index, self.ids, self.vectors = self.load_index()

        if self.index is None or self.ids is None:
            logger.error("Failed to load or build index")
            return []

        # Convert feature vector to numpy array
        query_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1.0
        query_vector = query_vector / query_norm

        # Search the index
        k = min(num, len(self.ids))
        if k == 0:
            return []

        # Get distances and indices
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=k)

        # Convert distances to similarities (1 - distance)
        similarities = [1 - dist for dist in distances[0]]

        # Map indices to database IDs
        results = [(self.ids[idx], sim) for idx, sim in zip(indices[0], similarities)]

        return results

    def rebuild_index(self):
        """
        Force rebuild the index regardless of whether it exists.

        Returns:
            tuple: (index, ids, vectors) for the new index.
        """
        if self.index_file.exists():
            self.index_file.unlink()
        if self.id_map_file.exists():
            self.id_map_file.unlink()

        return self.build_index()

if __name__ == "__main__":
    # Build index when run directly
    index = VectorIndex()
    index.build_index()