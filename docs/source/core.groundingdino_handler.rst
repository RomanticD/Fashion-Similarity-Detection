core.groundingdino_handler
==========================

.. automodule:: core.groundingdino_handler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Module Description
------------------

This module is primarily used to detect clothing regions in input images and conduct similarity comparisons on segmented images. It encompasses functions such as model loading, image segmentation, clothing detection, and file cleaning.

Functions
---------

.. autofunction:: core.groundingdino_handler.detect_clothes_in_image

   This function detects clothing regions in the input image and returns the bounding boxes of the detected clothes. It loads the GroundingDINO model, splits the input image into segments, performs inference on each segment to detect clothing, and returns a list of bounding boxes for the detected clothing regions. Returns an empty list if no clothes are detected or an error occurs.

   **Parameters**:
      - ``image (np.ndarray)``: The input image represented as a numpy array, typically obtained by converting a PIL image.
      - ``FRAME_WINDOW (optional)``: A Streamlit window object used to display the processed image during the detection process.

   **Returns**:
      - ``list``: A list of bounding boxes (as numpy arrays) for the detected clothing regions. Returns an empty list if no clothes are detected or an error occurs.

   **Exceptions**:
      - ``FileNotFoundError``: Raised if the weights file is not found.
      - ``Exception``: Raised if any error occurs during image processing or inference.

.. autofunction:: core.groundingdino_handler.clear_directory

   This function clears all files and empty folders within a specified directory.

   **Parameters**:
      - ``data_dir (Path)``: The path of the directory to be cleared.

   **Exceptions**:
      - Prints a message if `data_dir` is not a valid directory.