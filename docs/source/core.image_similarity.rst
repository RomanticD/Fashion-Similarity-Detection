core.image_similarity
=====================

.. automodule:: core.image_similarity
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Module Description
------------------

This module is dedicated to image feature extraction and similarity comparison in the image analysis pipeline. It loads a pre - trained ResNet50 model, configures pre - processing transforms, and offers functions to extract features from various input types and compare image similarities.

Functions
---------

.. autofunction:: core.image_similarity.extract_feature

   Extracts features from an image. It supports multiple input formats, including file paths, PIL images, and NumPy arrays.

   **Parameters**:
      - ``img_input``: The input image, which can be a file path, a PIL image object, or a NumPy array.

   **Returns**:
      - ``np.ndarray``: The extracted feature vector.

   **Exceptions**:
      - ``ValueError``: Raised if the input type is not supported.

.. autofunction:: core.image_similarity.load_images_from_arrays

   Loads features from in - memory image arrays.

   **Parameters**:
      - ``image_arrays (list)``: A list of NumPy arrays representing images.

   **Returns**:
      - ``dict``: A dictionary where keys are segment names and values are feature vectors.

.. autofunction:: core.image_similarity.load_single_image_feature_vector

   Loads the feature vector of a single image.

   **Parameters**:
      - ``img_path (str or Path)``: The file path of the image.

   **Returns**:
      - ``dict``: A dictionary where the key is the image name and the value is the feature vector.

.. autofunction:: core.image_similarity.cosine_similarity

   Calculates the cosine similarity between two vectors.

   **Parameters**:
      - ``vec1 (np.ndarray)``: The first vector.
      - ``vec2 (np.ndarray)``: The second vector.

   **Returns**:
      - ``float``: The cosine similarity value.

.. autofunction:: core.image_similarity.compare_similarities

   Compares the similarity between a single image and multiple images.

   **Parameters**:
      - ``single_dict (dict)``: A dictionary containing the feature vector of a single image.
      - ``images_dict (dict)``: A dictionary containing the feature vectors of multiple images.

   **Returns**:
      - ``list``: A list of tuples, where each tuple contains the image name and its similarity score.