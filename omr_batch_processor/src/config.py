# /omr_batch_processor/src/config.py
"""
Configuration constants for the OMR Grader application.
"""
import os
import cv2

# --- Core Paths ---
# Base directory is one level up from the 'src' directory where this file lives
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUT_DIR = os.path.join(BASE_DIR, 'omr_input')
OUTPUT_VISUAL_DIR = os.path.join(BASE_DIR, 'graded_output')
CSV_DIR = os.path.join(BASE_DIR, 'csv_data')

MASTER_ANSWERS_PATH = os.path.join(CSV_DIR, 'master_answers.csv')
STUDENT_RESULTS_DIR = os.path.join(CSV_DIR, 'student_results')


# --- Bubble Detection Parameters ---
BUBBLE_MIN_AREA = 150
BUBBLE_MAX_AREA = 2500
BUBBLE_MIN_CIRCULARITY = 0.7
BUBBLE_ASPECT_RATIO_RANGE = (0.7, 1.3)
CORNER_MARKER_MARGIN = 100 # Pixels from edge to ignore as registration marks


# --- Grid Layout Parameters ---
TOTAL_ROWS = 16
TOTAL_COLUMNS_PER_ROW = 15 # e.g., 3 questions * 5 choices
CHOICES_PER_QUESTION = 5
TOTAL_QUESTIONS = 48


# --- Answer Extraction Parameters ---
# The minimum difference in fill score between the most-filled and second-most-filled
# bubble to count as a valid answer. Helps detect multiple answers.
FILL_SCORE_CONFIDENCE_THRESHOLD = 20
# Any bubble with a fill score below this value is considered empty.
FILL_SCORE_MIN_THRESHOLD = 20


# --- Visualization Parameters ---
VIS_ROW_OVERLAY_COLOR = (255, 0, 0)      # Blue for row highlight
VIS_CORRECT_ANSWER_COLOR = (0, 255, 0)   # Green
VIS_WRONG_ANSWER_COLOR = (0, 0, 255)     # Red
VIS_DEFAULT_BUBBLE_COLOR = (255, 0, 0) # Blue for all detected bubbles
VIS_TEXT_COLOR = (0, 0, 0)               # Black
VIS_ROW_OVERLAY_ALPHA = 0.15
VIS_THICKNESS_BUBBLE = 2
VIS_THICKNESS_ANSWER = 4

# --- Visualization Parameters for Header Info ---
VIS_INFO_FONT = cv2.FONT_HERSHEY_SIMPLEX
VIS_INFO_FONT_SCALE = 0.9
VIS_INFO_FONT_THICKNESS = 2

# Vertical position for the Name line (higher on the page)
VIS_NAME_Y_OFFSET = -80

# Vertical position for the Score line (lower on the page)
VIS_SCORE_Y_OFFSET = -55

# Horizontal position offsets
VIS_NAME_X_OFFSET = 160
VIS_SCORE_X_OFFSET = -30