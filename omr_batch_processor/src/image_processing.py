# image_processing.py
"""
Functions for loading and preprocessing images for OMR.
"""
import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image at {image_path}")
    print(f"Loaded image: {image_path} (shape={image.shape})")
    return image

def preprocess_for_omr(image):
    """
    Converts an image to grayscale and then binarizes it using Otsu's threshold.
    Returns the grayscale and binarized images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Converted to grayscale. Shape: {gray.shape}")

    # Binarize using Otsu's method, inverting the image so bubbles are white
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("Binarized image using Otsu's threshold.")
    
    # Use a morphological closing operation to fill small holes in bubbles
    kernel = np.ones((3, 3), np.uint8)
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    return gray, binarized