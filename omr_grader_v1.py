# omr_grader_complete.py
import cv2
import numpy as np
import os
import csv
import pandas as pd
import sys

# --- Configuration Constants ---
CANNY_LOWER_THRESH = 50
CANNY_UPPER_THRESH = 150
WARPED_IMAGE_WIDTH = 1000
WARPED_IMAGE_HEIGHT = 1414
COLUMN_PROJECTION_THRESH_FACTOR = 0.02   # Much more sensitive
ROW_PROJECTION_THRESH_FACTOR = 0.03      # Much more sensitive
MIN_GAP_BETWEEN_BLOCKS = 15              # Smaller gaps
BUBBLE_MIN_AREA = 80                     # Even smaller bubbles
BUBBLE_MAX_AREA = 3000                   # Larger maximum
BUBBLE_MIN_CIRCULARITY = 0.3             # Very lenient
CHOICES_PER_QUESTION = 5
MIN_FILLED_PIXEL_COUNT = 50              # Lower threshold
AMBIGUITY_RATIO_THRESH = 0.6

class OMRGrader:
    """
    A robust, scalable OMR grading class that dynamically discovers the layout
    of an answer sheet.
    """
    def __init__(self):
        """
        Initializes the grader.
        """
        self.answer_labels = ['A', 'B', 'C', 'D', 'E']

    def _get_perspective_transform(self, image):
        """
        Applies perspective transformation to correct the image orientation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, CANNY_LOWER_THRESH, CANNY_UPPER_THRESH)

        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Warning: No contours found. Using original image.")
            return image

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        approx = None
        for c in contours:
            # Skip very small contours
            if cv2.contourArea(c) < 10000:
                continue
                
            perimeter = cv2.arcLength(c, True)
            # Try different approximation factors
            for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
                candidate = cv2.approxPolyDP(c, epsilon_factor * perimeter, True)
                if len(candidate) == 4:
                    approx = candidate
                    break
            if approx is not None:
                break

        if approx is None:
            print("Warning: Could not find 4-corner contours. Using original image.")
            return image

        # Order the points for a consistent transformation
        points = approx.reshape(4, 2)
        ordered_points = np.zeros((4, 2), dtype="float32")
        
        # Top-left point has the smallest sum
        # Bottom-right has the largest sum
        s = points.sum(axis=1)
        ordered_points[0] = points[np.argmin(s)]
        ordered_points[2] = points[np.argmax(s)]
        
        # Top-right point has the smallest difference
        # Bottom-left has the largest difference
        diff = np.diff(points, axis=1)
        ordered_points[1] = points[np.argmin(diff)]
        ordered_points[3] = points[np.argmax(diff)]

        dest_points = np.array([
            [0, 0],
            [WARPED_IMAGE_WIDTH - 1, 0],
            [WARPED_IMAGE_WIDTH - 1, WARPED_IMAGE_HEIGHT - 1],
            [0, WARPED_IMAGE_HEIGHT - 1]], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_points, dest_points)
        warped = cv2.warpPerspective(image, matrix, (WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT))
        
        return warped

    def _find_content_blocks(self, projection, threshold_factor, min_gap):
        """
        Analyzes a 1D projection profile to find contiguous blocks of content.
        """
        if len(projection) == 0 or np.max(projection) == 0:
            return []
        
        # Smooth the projection to reduce noise
        smoothed = cv2.GaussianBlur(projection.astype(np.float32), (5, 1), 0).flatten()
        
        # Use multiple threshold strategies and combine results
        thresholds = [
            np.max(smoothed) * threshold_factor,
            np.mean(smoothed) + np.std(smoothed),
            np.percentile(smoothed, 75)
        ]
        
        all_blocks = []
        
        for threshold in thresholds:
            # Create binary projection
            binary_proj = (smoothed > threshold).astype(np.uint8)
            
            # Find contiguous regions using connected components
            blocks = []
            in_block = False
            start = 0
            
            for i, val in enumerate(binary_proj):
                if val == 1 and not in_block:
                    start = i
                    in_block = True
                elif val == 0 and in_block:
                    # Check if gap is large enough to end block
                    gap_start = i
                    gap_size = 0
                    
                    # Look ahead to see gap size
                    for j in range(i, min(i + min_gap, len(binary_proj))):
                        if binary_proj[j] == 0:
                            gap_size += 1
                        else:
                            break
                    
                    if gap_size >= min_gap:
                        blocks.append((start, i))
                        in_block = False
                        
            if in_block:
                blocks.append((start, len(binary_proj)))
            
            # Filter blocks by minimum size
            valid_blocks = [(start, end) for start, end in blocks if end - start > 15]
            all_blocks.extend(valid_blocks)
        
        # Merge overlapping blocks and remove duplicates
        if not all_blocks:
            return []
        
        # Sort blocks by start position
        all_blocks.sort(key=lambda x: x[0])
        
        # Merge overlapping blocks
        merged_blocks = [all_blocks[0]]
        for current_start, current_end in all_blocks[1:]:
            last_start, last_end = merged_blocks[-1]
            
            # If blocks overlap or are very close, merge them
            if current_start <= last_end + 10:
                merged_blocks[-1] = (last_start, max(last_end, current_end))
            else:
                merged_blocks.append((current_start, current_end))
        
        return merged_blocks

    def _discover_question_rows(self, binarized_warped_image):
        """
        Dynamically finds the ROIs for each question row on the sheet.
        """
        height, width = binarized_warped_image.shape
        
        # Method 1: Column-based detection
        vertical_projection = np.sum(binarized_warped_image, axis=0)
        column_blocks = self._find_content_blocks(vertical_projection, 
                                                COLUMN_PROJECTION_THRESH_FACTOR, 
                                                MIN_GAP_BETWEEN_BLOCKS)
        
        question_rows = []
        
        # If no clear columns found, try full-width analysis
        if not column_blocks or len(column_blocks) < 1:
            print("No clear columns detected, using full-width analysis")
            column_blocks = [(50, width - 50)]  # Use most of the width, avoid edges
        
        print(f"Found {len(column_blocks)} column blocks: {column_blocks}")
        
        # For each column, find horizontal blocks (rows)
        for col_idx, (col_start, col_end) in enumerate(column_blocks):
            col_start = max(0, col_start)
            col_end = min(width, col_end)
            
            # Extract the column region
            column_roi = binarized_warped_image[:, col_start:col_end]
            
            # Calculate horizontal projection for this column
            horizontal_projection = np.sum(column_roi, axis=1)
            
            # Find row blocks within this column
            row_blocks = self._find_content_blocks(horizontal_projection, 
                                                 ROW_PROJECTION_THRESH_FACTOR, 
                                                 MIN_GAP_BETWEEN_BLOCKS)
            
            print(f"Column {col_idx+1} ({col_start}-{col_end}): Found {len(row_blocks)} row blocks")
            
            # Convert row blocks to question row ROIs
            for row_idx, (row_start, row_end) in enumerate(row_blocks):
                roi_width = col_end - col_start
                roi_height = row_end - row_start
                
                # More lenient filtering - accept various sizes
                if (roi_height >= 15 and roi_height <= 300 and 
                    roi_width >= 50 and roi_width <= width):
                    
                    question_rows.append((col_start, row_start, roi_width, roi_height))
                    print(f"  Added row {row_idx+1}: x={col_start}, y={row_start}, w={roi_width}, h={roi_height}")
        
        # Method 2: If we have very few rows, try a different approach
        if len(question_rows) < 5:
            print("Too few rows detected, trying alternative method...")
            
            # Try horizontal scanning with different parameters
            horizontal_projection = np.sum(binarized_warped_image, axis=1)
            
            # Use a lower threshold for row detection
            alt_row_blocks = self._find_content_blocks(horizontal_projection, 0.03, 10)
            
            print(f"Alternative method found {len(alt_row_blocks)} potential rows")
            
            # Add rows that weren't found before
            for row_start, row_end in alt_row_blocks:
                roi_height = row_end - row_start
                
                # Check if this row overlaps with existing rows
                overlaps = False
                for existing_x, existing_y, existing_w, existing_h in question_rows:
                    if (row_start < existing_y + existing_h and 
                        row_end > existing_y):
                        overlaps = True
                        break
                
                if not overlaps and roi_height >= 15 and roi_height <= 200:
                    # Use a reasonable width
                    roi_width = min(width - 100, 800)
                    roi_x = max(50, (width - roi_width) // 2)
                    
                    question_rows.append((roi_x, row_start, roi_width, roi_height))
                    print(f"  Added alternative row: x={roi_x}, y={row_start}, w={roi_width}, h={roi_height}")
        
        # Method 3: If still too few rows, use grid-based estimation
        if len(question_rows) < 10:
            print("Still too few rows, using grid-based estimation...")
            
            # Estimate based on typical answer sheet layout
            # Assume questions are distributed evenly in the middle portion of the sheet
            start_y = height // 6  # Skip header area
            end_y = height - height // 6  # Skip footer area
            
            # Look for content in this region
            content_region = binarized_warped_image[start_y:end_y, :]
            content_projection = np.sum(content_region, axis=1)
            
            # Find peaks in the projection (likely question rows)
            from scipy import signal
            try:
                # Find peaks with minimum distance between them
                peaks, _ = signal.find_peaks(content_projection, 
                                           height=np.percentile(content_projection, 60),
                                           distance=20)
                
                print(f"Found {len(peaks)} peaks in content region")
                
                for peak in peaks:
                    actual_y = start_y + peak
                    
                    # Check if this peak area has reasonable content
                    roi_y_start = max(0, actual_y - 15)
                    roi_y_end = min(height, actual_y + 35)
                    roi_height = roi_y_end - roi_y_start
                    
                    if roi_height >= 20:
                        roi_width = min(width - 100, 800)
                        roi_x = max(50, (width - roi_width) // 2)
                        
                        # Check for overlap with existing rows
                        overlaps = False
                        for existing_x, existing_y, existing_w, existing_h in question_rows:
                            if (roi_y_start < existing_y + existing_h and 
                                roi_y_end > existing_y):
                                overlaps = True
                                break
                        
                        if not overlaps:
                            question_rows.append((roi_x, roi_y_start, roi_width, roi_height))
                            print(f"  Added grid-based row: x={roi_x}, y={roi_y_start}, w={roi_width}, h={roi_height}")
                            
            except ImportError:
                print("scipy not available, using simple grid estimation")
                # Fallback: create a simple grid
                estimated_rows = min(20, max(10, len(question_rows) * 2))
                row_height = (end_y - start_y) // estimated_rows
                
                for i in range(estimated_rows):
                    roi_y = start_y + i * row_height
                    roi_width = min(width - 100, 800)
                    roi_x = max(50, (width - roi_width) // 2)
                    
                    if roi_y + row_height < height:
                        question_rows.append((roi_x, roi_y, roi_width, row_height))
        
        # Sort by vertical position first, then horizontal position
        question_rows.sort(key=lambda r: (r[1], r[0]))
        
        # Remove very similar rows (potential duplicates)
        if len(question_rows) > 1:
            filtered_rows = [question_rows[0]]
            for current_row in question_rows[1:]:
                last_row = filtered_rows[-1]
                
                # If rows are very close vertically, keep only one
                y_diff = abs(current_row[1] - last_row[1])
                if y_diff > 10:  # Minimum vertical separation
                    filtered_rows.append(current_row)
            
            question_rows = filtered_rows
        
        print(f"Final result: {len(question_rows)} question rows detected")
        return question_rows

    def _is_bubble(self, contour):
        """
        Checks if a given contour has the geometric properties of an answer bubble.
        """
        area = cv2.contourArea(contour)
        if not (BUBBLE_MIN_AREA < area < BUBBLE_MAX_AREA):
            return False

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < BUBBLE_MIN_CIRCULARITY:
            return False
        
        # Additional check: aspect ratio should be close to 1 (circular)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        return True

    def _analyze_bubbles_in_roi(self, question_roi_binary, question_roi_color, debug=False):
        """
        Analyzes bubbles within a question ROI and returns the selected answer.
        """
        # Find contours in the ROI
        contours, _ = cv2.findContours(question_roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find bubbles
        bubbles = [c for c in contours if self._is_bubble(c)]
        
        if debug:
            print(f"    Found {len(bubbles)} potential bubbles")
        
        # If we don't have exactly 5 bubbles, try different approaches
        if len(bubbles) != CHOICES_PER_QUESTION:
            # Try with more lenient criteria
            lenient_bubbles = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 50:  # Very lenient area check
                    lenient_bubbles.append(c)
            
            if len(lenient_bubbles) >= CHOICES_PER_QUESTION:
                # Sort by area and take the largest ones that might be bubbles
                bubbles = sorted(lenient_bubbles, key=cv2.contourArea, reverse=True)[:CHOICES_PER_QUESTION]
            else:
                return "Invalid_Layout"
        
        # Sort bubbles by horizontal position (left to right)
        bubbles.sort(key=lambda b: cv2.boundingRect(b)[0])
        
        # Analyze fill level of each bubble
        pixel_counts = []
        for bubble_contour in bubbles:
            # Create a mask for this bubble
            mask = np.zeros(question_roi_binary.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble_contour], -1, 255, -1)
            
            # Apply the mask to the binary ROI
            masked_roi = cv2.bitwise_and(question_roi_binary, question_roi_binary, mask=mask)
            pixel_count = cv2.countNonZero(masked_roi)
            pixel_counts.append(pixel_count)
        
        if debug:
            print(f"    Pixel counts: {pixel_counts}")
        
        # Determine the answer based on pixel counts
        if not pixel_counts:
            return "No_Bubbles"
        
        max_pixels = max(pixel_counts)
        
        # Check if any bubble is sufficiently filled
        if max_pixels < MIN_FILLED_PIXEL_COUNT:
            return "Skipped"
        
        # Check for ambiguous answers (multiple bubbles filled)
        sorted_counts = sorted(pixel_counts, reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[1] / sorted_counts[0] > AMBIGUITY_RATIO_THRESH:
            return "Invalid"
        
        # Return the answer corresponding to the most filled bubble
        marked_index = np.argmax(pixel_counts)
        return self.answer_labels[marked_index]

    def process_sheet(self, image_path, debug=False):
        """
        Main processing function. Converts a scanned image to a dictionary of answers.
        """
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"error": f"Could not read image: {image_path}"}

        print(f"Original image shape: {original_image.shape}")

        # --- Phase 1: Perspective Correction ---
        warped_color = self._get_perspective_transform(original_image)
        
        if debug:
            cv2.imshow("Step 1: Warped Image", cv2.resize(warped_color, (500, 707)))
            cv2.waitKey(0)
        
        # Convert to grayscale and binarize
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        
        # Use multiple binarization techniques and combine them
        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(warped_gray, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 15, 7)
        
        # Otsu's threshold
        _, otsu_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine both thresholds
        binarized = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

        if debug:
            cv2.imshow("Step 2: Binarized Image", cv2.resize(binarized, (500, 707)))
            cv2.waitKey(0)

        # --- Phase 2: Layout Discovery ---
        question_row_rois = self._discover_question_rows(binarized)
        
        print(f"Found {len(question_row_rois)} potential question rows")
        
        if debug:
            debug_img = warped_color.copy()
            for i, (x, y, w, h) in enumerate(question_row_rois):
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Step 3: Discovered Question Rows", cv2.resize(debug_img, (500, 707)))
            cv2.waitKey(0)

        # --- Phase 3: Bubble Analysis ---
        student_answers = {}
        
        for i, (x, y, w, h) in enumerate(question_row_rois):
            q_num = i + 1
            print(f"Processing question {q_num}...")
            
            # Extract the ROI
            question_roi_binary = binarized[y:y+h, x:x+w]
            question_roi_color = warped_color[y:y+h, x:x+w]
            
            # Analyze bubbles in this ROI
            answer = self._analyze_bubbles_in_roi(question_roi_binary, question_roi_color, debug)
            
            if debug:
                print(f"Question {q_num}: {answer}")
                roi_display = cv2.resize(question_roi_color, (400, 60))
                cv2.imshow(f"Question {q_num} - Answer: {answer}", roi_display)
                cv2.waitKey(0)
            
            student_answers[q_num] = answer
        
        if debug:
            cv2.destroyAllWindows()
            
        return student_answers

    def save_answers_to_csv(self, answers_dict, output_path):
        """
        Saves the answers dictionary to a CSV file.
        """
        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Question', 'Answer'])
                for q_num, answer in sorted(answers_dict.items()):
                    writer.writerow([q_num, answer])
            return True
        except IOError:
            print(f"Error: Could not write to file '{output_path}'")
            return False

def grade_from_csv(student_csv_path, master_csv_path):
    """
    Compares a student's answer CSV with a master answer key CSV and calculates the score.
    """
    try:
        student_df = pd.read_csv(student_csv_path)
        master_df = pd.read_csv(master_csv_path)
    except FileNotFoundError as e:
        return {"error": f"File not found: {e.filename}"}
   
    # Data validation
    required_cols = ['Question', 'Answer']
    if not all(col in student_df.columns for col in required_cols):
        return {"error": f"Student CSV is missing required columns. Found: {student_df.columns.tolist()}"}
    if not all(col in master_df.columns for col in required_cols):
        return {"error": f"Master CSV is missing required columns. Found: {master_df.columns.tolist()}"}
       
    # Merge and compare
    comparison_df = pd.merge(student_df, master_df, on='Question', suffixes=('_student', '_master'))
    comparison_df['is_correct'] = (comparison_df['Answer_student'] == comparison_df['Answer_master'])
   
    score = comparison_df['is_correct'].sum()
    total_questions = len(master_df)
    
    return {
        "score": int(score),
        "total_questions": total_questions,
        "comparison_details": comparison_df
    }

def main():
    """
    Main function to process a student's sheet and optionally compare with answer key.
    """
    # Configuration
    image_to_process = 'student_sheet_filled.png'
    student_answers_csv = 'student_answers.csv'
    master_answers_csv = 'master_answers.csv'
    
    # Initialize grader
    grader = OMRGrader()
    
    # Process the sheet
    print(f"Processing '{image_to_process}'...")
    student_answers = grader.process_sheet(image_to_process, debug=True)
    
    if "error" in student_answers:
        print(f"Error: {student_answers['error']}")
        return
    
    print(f"Successfully processed {len(student_answers)} questions")
    
    # Save student answers
    if grader.save_answers_to_csv(student_answers, student_answers_csv):
        print(f"Student answers saved to '{student_answers_csv}'")
    
    # Compare with master answers if available
    if os.path.exists(master_answers_csv):
        print(f"\nComparing with master answer key...")
        results = grade_from_csv(student_answers_csv, master_answers_csv)
        
        if "error" in results:
            print(f"Grading error: {results['error']}")
        else:
            print(f"\n--- GRADING RESULTS ---")
            print(f"Score: {results['score']} / {results['total_questions']}")
            print(f"Percentage: {(results['score'] / results['total_questions']) * 100:.1f}%")
            
            print(f"\n--- DETAILED BREAKDOWN ---")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)
            print(results['comparison_details'][['Question', 'Answer_student', 'Answer_master', 'is_correct']])
    else:
        print(f"Master answer key '{master_answers_csv}' not found. Skipping comparison.")

if __name__ == '__main__':
    main()