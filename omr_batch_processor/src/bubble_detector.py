# bubble_detector.py
"""
Functions to detect and validate answer bubbles from a binarized image.
"""
import cv2
from . import config
import numpy as np

def _is_valid_bubble(contour):
    """
    Determines if a contour represents a valid bubble based on geometric properties.
    """
    area = cv2.contourArea(contour)
    if not (config.BUBBLE_MIN_AREA < area < config.BUBBLE_MAX_AREA):
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < config.BUBBLE_MIN_CIRCULARITY:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    min_ratio, max_ratio = config.BUBBLE_ASPECT_RATIO_RANGE
    if not (min_ratio <= aspect_ratio <= max_ratio):
        return False

    return True

def _is_in_corner(cx, cy, img_shape):
    """Checks if a point is in a corner, likely a registration mark."""
    h, w = img_shape[:2]
    margin = config.CORNER_MARKER_MARGIN
    return (
        (cx < margin and cy < margin) or
        (cx > w - margin and cy < margin) or
        (cx < margin and cy > h - margin) or
        (cx > w - margin and cy > h - margin)
    )

def find_bubbles(binarized_image, original_image_shape):
    """
    Finds all contours in a binarized image and filters them to find bubbles.
    Returns a list of bubble dictionaries.
    """
    contours, hierarchy = cv2.findContours(binarized_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    bubbles = []
    if hierarchy is None:
        return bubbles

    # We only want parent contours (no contour inside another)
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1 and _is_valid_bubble(contour):
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Skip bubbles that are likely corner registration marks
                if not _is_in_corner(cx, cy, original_image_shape):
                    bubbles.append({'cx': cx, 'cy': cy, 'contour': contour})

    print(f"Filtered to {len(bubbles)} potential bubbles.")
    return bubbles