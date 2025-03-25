core.image_processing
=====================

.. automodule:: core.image_processing
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Module Description
------------------

This module provides a set of auxiliary image - processing functions. These functions are used for operations such as image segmentation, stitching, padding, inference, and pre - processing, which help the GroundingDINO model achieve more precise identification in long web screenshots.

Functions
---------

.. autofunction:: core.image_processing.split_image_vertically

   Vertically splits an image into multiple segments, each with a height of `segment_height`.

   **Parameters**:
      - ``image (np.ndarray)``: The input image with the shape (height, width, channels).
      - ``segment_height (int)``: The height of each segmented image.

   **Returns**:
      - ``List[np.ndarray]``: A list of segmented images.

   **Exceptions**:
      - ``ValueError``: Raised if `segment_height` is greater than the image height.

.. autofunction:: core.image_processing.combine_segments_vertically

   Re - combines the segmented image segments into an image of the original size.

   **Parameters**:
      - ``segments (List[np.ndarray])``: A list of segmented image segments.
      - ``original_height (int)``: The height of the original image.
      - ``original_width (int)``: The width of the original image.

   **Returns**:
      - ``np.ndarray``: The combined image.

.. autofunction:: core.image_processing.pad_image

   Pads or resizes an image to the target size and centers it.

   **Parameters**:
      - ``image (Image.Image)``: The input image.
      - ``target_size (Tuple[int, int])``: The target size of the image (width, height).

   **Returns**:
      - ``Image.Image``: The padded image.

.. autofunction:: core.image_processing.run_inference

   Performs inference on each image segment, detects targets, and annotates them.

   **Parameters**:
      - ``model``: The model used for inference.
      - ``transform``: The image transformation function.
      - ``segments (List[np.ndarray])``: A list of input image segments.
      - ``TEXT_PROMPT (str)``: The text prompt for target detection.
      - ``BOX_THRESHOLD (float)``: The confidence threshold for the bounding box.
      - ``FRAME_WINDOW``: The window object used to display the results.

   **Returns**:
      - ``List[np.ndarray]``: A list of all target - detected image segments.

.. autofunction:: core.image_processing.prepare_transform

   Prepares the pre - processing function for image transformation.

   **Returns**:
      - ``T.Compose``: A transformation object composed of multiple image processing steps.