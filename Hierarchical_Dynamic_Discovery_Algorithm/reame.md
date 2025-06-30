# OMR Sheet Checker - Hierarchical Dynamic Discovery Algorithm

This project provides an Optical Mark Recognition (OMR) solution for grading multiple-choice answer sheets using computer vision. It detects filled bubbles, extracts student answers, compares them to a master answer key, and visualizes the results.

## Features
- Automatic detection of answer bubbles using contour analysis
- Robust to empty/filled bubbles and QR code/marker exclusion
- Groups bubbles into rows and questions dynamically
- Extracts student answers and compares with master answers
- Generates a CSV report and annotated output image
- Visualizes correct and incorrect answers

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies (if needed):
```bash
pip install opencv-python numpy pandas matplotlib scipy
```

## Usage

### 1. Prepare Input Files
- Place the scanned/fillable OMR sheet image in the project directory (default: `student_sheet_filled.png`).
- Prepare a master answer key as a CSV file named `master_answers.csv` with columns:
  - `Question` (1-based question number)
  - `Answer` (A/B/C/D/E)

Example `master_answers.csv`:
```
Question,Answer
1,B
2,D
3,A
...
```

### 2. Run the Grader
```bash
python omr_grader.py [path_to_image]
```
- If no image path is provided, it uses `student_sheet_filled.png` by default.

### 3. Output Files
- `student_answers.csv`: Contains extracted answers, comparison with master, and scores.
- `omr_graded_output.png`: Annotated image showing detected bubbles, selected answers, and question numbers.

## Algorithm Overview

1. **Image Preprocessing**
   - Convert input image to grayscale.
   - Apply Otsu's thresholding to binarize (invert: filled = white).
   - Morphological closing to fill small gaps.

2. **Bubble Detection**
   - Find contours using OpenCV's `findContours` with hierarchy.
   - Filter contours by area, circularity, and aspect ratio to identify bubbles.
   - Exclude bubbles in the corners (to avoid QR codes/markers).

3. **Bubble Grouping**
   - Group bubbles into rows using y-coordinate percentiles.
   - Within each row, sort bubbles by x-coordinate.
   - Assign each bubble to a question and choice (A-E) based on its position.

4. **Answer Extraction**
   - For each question, calculate the mean pixel value inside each bubble (higher = more filled).
   - Select the most filled bubble as the answer, if it is clearly more filled than the others.

5. **Scoring and Output**
   - Compare extracted answers to the master answer key.
   - Write results and scores to `student_answers.csv`.
   - Visualize results on the image, highlighting correct/incorrect answers.

## Customization
- Adjust bubble detection parameters in `omr_grader.py` for different sheet layouts:
  - `BUBBLE_MIN_AREA`, `BUBBLE_MAX_AREA`, `BUBBLE_MIN_CIRCULARITY`, `BUBBLE_ASPECT_RATIO_RANGE`
- Tune the fill score threshold for ambiguous answers.

## Troubleshooting
- Ensure the input image is clear and well-aligned.
- If bubbles are missed or false positives occur, adjust detection parameters.
- Check the console output for row/question/bubble detection diagnostics.

## License
