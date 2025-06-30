# fixed_omr_grader_48.py
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
BUBBLE_MIN_AREA = 80
BUBBLE_MAX_AREA = 3000
BUBBLE_MIN_CIRCULARITY = 0.3
CHOICES_PER_QUESTION = 5
MIN_FILLED_PIXEL_COUNT = 50
AMBIGUITY_RATIO_THRESH = 0.6

# Expected questions per column for 48-question layout
EXPECTED_QUESTIONS_PER_COLUMN = 16
TOTAL_EXPECTED_QUESTIONS = 48

class FixedOMRGrader:
    """
    Fixed OMR grader that properly handles 3-column layout with exactly 48 questions (16 per column).
    """
    def __init__(self):
        self.answer_labels = ['A', 'B', 'C', 'D', 'E']

    def _get_perspective_transform(self, image):
        """
        Applies perspective transformation to correct the image orientation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, CANNY_LOWER_THRESH, CANNY_UPPER_THRESH)

        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Warning: No contours found. Using original image.")
            return image

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        approx = None
        for c in contours:
            if cv2.contourArea(c) < 10000:
                continue
                
            perimeter = cv2.arcLength(c, True)
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

        # Order the points for perspective transformation
        points = approx.reshape(4, 2)
        ordered_points = np.zeros((4, 2), dtype="float32")
        
        s = points.sum(axis=1)
        ordered_points[0] = points[np.argmin(s)]
        ordered_points[2] = points[np.argmax(s)]
        
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

    def _find_answer_area(self, binarized_image):
        """
        Identifies the main answer area and excludes header/footer regions.
        """
        height, width = binarized_image.shape
        
        # Skip top 15% (header with instructions) and bottom 10% (footer)
        start_y = int(height * 0.15)
        end_y = int(height * 0.90)
        
        # Skip left and right margins (5% each)
        start_x = int(width * 0.05)
        end_x = int(width * 0.95)
        
        answer_area = binarized_image[start_y:end_y, start_x:end_x]
        
        print(f"Answer area extracted: {start_x}-{end_x} (width), {start_y}-{end_y} (height)")
        return answer_area, (start_x, start_y)

    def _detect_columns(self, answer_area):
        """
        Detects the 3 distinct columns by finding the two main vertical separators,
        while ignoring spurious gaps like the one between question numbers and bubbles.
        """
        height, width = answer_area.shape
        vertical_projection = np.sum(answer_area, axis=0)
        smoothed = cv2.GaussianBlur(vertical_projection.astype(np.float32), (15, 1), 0).flatten()

        # Find ALL local minima (valleys) which are potential separators
        all_valleys = []
        threshold = np.max(smoothed) * 0.5 
        for i in range(20, len(smoothed) - 20):
            window = smoothed[i-15:i+16]
            if smoothed[i] < threshold and smoothed[i] == np.min(window):
                 all_valleys.append(i)

        if not all_valleys:
            print("Warning: No valleys found. Falling back to equal width division.")
            col_width = width // 3
            return [(0, col_width), (col_width, 2*col_width), (2*col_width, width)]
            
        # Group close valleys and find the deepest point in each group
        grouped_valleys = []
        current_group = [all_valleys[0]]
        for valley in all_valleys[1:]:
            if valley - current_group[-1] < 50:
                current_group.append(valley)
            else:
                deepest = min(current_group, key=lambda x: smoothed[x])
                grouped_valleys.append(deepest)
                current_group = [valley]
        deepest = min(current_group, key=lambda x: smoothed[x])
        grouped_valleys.append(deepest)
        
        print(f"Found {len(grouped_valleys)} distinct valley locations: {grouped_valleys}")

        # Filter out valleys too close to the edges. This is key to ignoring the
        # gap between the question numbers (1,2,3..) and the first bubble column.
        min_dist_from_edge = int(width * 0.15)
        candidate_valleys = [v for v in grouped_valleys if min_dist_from_edge < v < (width - min_dist_from_edge)]
        
        print(f"Filtered to {len(candidate_valleys)} candidate valleys away from edges: {candidate_valleys}")

        columns = []
        if len(candidate_valleys) >= 2:
            # If we have multiple candidates, pick the two deepest (lowest projection value)
            candidate_valleys.sort(key=lambda x: smoothed[x])
            separators = sorted(candidate_valleys[:2])
            
            # Define column boundaries based on the two separators
            columns.append((0, separators[0]))
            columns.append((separators[0], separators[1]))
            columns.append((separators[1], width))
        else:
            # If robust detection fails, fall back to a safe equal division.
            print(f"Warning: Could not identify two clear column separators. Falling back to equal width division.")
            col_width = width // 3
            columns = [
                (0, col_width),
                (col_width, 2 * col_width),
                (2 * col_width, width)
            ]
            
        print(f"Detected {len(columns)} columns: {columns}")
        return columns

    def _detect_rows_in_column(self, column_area, expected_rows=EXPECTED_QUESTIONS_PER_COLUMN):
        """
        Detects question rows within a single column, limiting to expected number.
        """
        height, width = column_area.shape
        
        # Calculate horizontal projection (sum along columns)
        horizontal_projection = np.sum(column_area, axis=1)
        
        # Smooth the projection less aggressively to preserve row detail
        smoothed = cv2.GaussianBlur(horizontal_projection.astype(np.float32), (7, 1), 0).flatten()
        
        # Use adaptive threshold based on projection statistics
        threshold = np.mean(smoothed) * 0.5
        
        # Find peaks (high points) that indicate question rows
        peaks = []
        min_peak_distance = height // (expected_rows + 5)  # Minimum distance between peaks
        
        for i in range(min_peak_distance, len(smoothed) - min_peak_distance):
            if (smoothed[i] > threshold and 
                smoothed[i] > smoothed[i-min_peak_distance//2] and 
                smoothed[i] > smoothed[i+min_peak_distance//2]):
                # Check if this peak is far enough from existing peaks
                too_close = False
                for existing_peak in peaks:
                    if abs(i - existing_peak) < min_peak_distance:
                        # Keep the higher peak
                        if smoothed[i] > smoothed[existing_peak]:
                            peaks.remove(existing_peak)
                        else:
                            too_close = True
                        break
                
                if not too_close:
                    peaks.append(i)
        
        # Sort peaks and limit to expected number
        peaks.sort()
        
        # If we have too many peaks, keep the strongest ones
        if len(peaks) > expected_rows:
            # Sort by peak strength and keep the top ones
            peak_strengths = [(peak, smoothed[peak]) for peak in peaks]
            peak_strengths.sort(key=lambda x: x[1], reverse=True)
            peaks = [peak for peak, _ in peak_strengths[:expected_rows]]
            peaks.sort()  # Sort by position again
        
        # If we have too few peaks, try to find more with relaxed criteria
        elif len(peaks) < expected_rows:
            # Try with lower threshold
            lower_threshold = np.mean(smoothed) * 0.3
            additional_peaks = []
            
            for i in range(min_peak_distance, len(smoothed) - min_peak_distance):
                if (smoothed[i] > lower_threshold and 
                    smoothed[i] > smoothed[i-min_peak_distance//3] and 
                    smoothed[i] > smoothed[i+min_peak_distance//3]):
                    # Check if this peak is far enough from existing peaks
                    too_close = False
                    for existing_peak in peaks + additional_peaks:
                        if abs(i - existing_peak) < min_peak_distance//2:
                            too_close = True
                            break
                    
                    if not too_close:
                        additional_peaks.append(i)
            
            # Add the strongest additional peaks
            if additional_peaks:
                additional_strengths = [(peak, smoothed[peak]) for peak in additional_peaks]
                additional_strengths.sort(key=lambda x: x[1], reverse=True)
                needed = min(expected_rows - len(peaks), len(additional_strengths))
                peaks.extend([peak for peak, _ in additional_strengths[:needed]])
                peaks.sort()
        
        # Convert peaks to row boundaries
        rows = []
        average_row_height = height // expected_rows
        
        for i, peak in enumerate(peaks):
            # Define row boundaries around each peak
            row_height = min(average_row_height, 50)  # Cap at 50 pixels
            row_start = max(0, peak - row_height//2)
            row_end = min(height, peak + row_height//2)
            
            # Adjust boundaries to avoid overlap with previous row
            if i > 0 and row_start < rows[-1][1]:
                row_start = rows[-1][1]
            
            # Adjust boundaries to avoid overlap with next row
            if i < len(peaks) - 1:
                next_peak = peaks[i + 1]
                midpoint = (peak + next_peak) // 2
                row_end = min(row_end, midpoint)
            
            if row_end > row_start + 15:  # Minimum row height of 15 pixels
                rows.append((row_start, row_end))
        
        return rows

    def _discover_question_layout(self, binarized_warped_image):
        """
        Discovers question layout by finding columns first, then rows within each column.
        """
        # Extract the main answer area (exclude headers/footers)
        answer_area, offset = self._find_answer_area(binarized_warped_image)
        offset_x, offset_y = offset
        
        # Detect the 3 columns
        columns = self._detect_columns(answer_area)
        
        question_rois = []
        question_num = 1
        
        # Process each column
        for col_idx, (col_start, col_end) in enumerate(columns):
            print(f"\nProcessing column {col_idx + 1}...")
            
            # Extract column area
            column_area = answer_area[:, col_start:col_end]
            
            # Detect rows within this column
            rows = self._detect_rows_in_column(column_area, EXPECTED_QUESTIONS_PER_COLUMN)
            
            print(f"Found {len(rows)} rows in column {col_idx + 1}")
            
            # Create ROIs for each row in this column
            for row_start, row_end in rows:
                # Adjust coordinates back to full image
                roi_x = offset_x + col_start
                roi_y = offset_y + row_start
                roi_width = col_end - col_start
                roi_height = row_end - row_start
                
                question_rois.append({
                    'question_num': question_num,
                    'roi': (roi_x, roi_y, roi_width, roi_height)
                })
                
                print(f"  Question {question_num}: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
                question_num += 1
        
        return question_rois

    def _is_bubble(self, contour):
        """
        Checks if a contour represents a valid bubble.
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
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        return True

    def _analyze_bubbles_in_roi(self, question_roi_binary, debug=False):
        """
        Analyzes bubbles within a question ROI.
        """
        contours, _ = cv2.findContours(question_roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = [c for c in contours if self._is_bubble(c)]
        
        if debug:
            print(f"    Found {len(bubbles)} valid bubbles")
        
        # Handle cases where we don't have exactly 5 bubbles
        if len(bubbles) < CHOICES_PER_QUESTION:
            # Try more lenient criteria
            lenient_bubbles = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 50:  # Very basic area check
                    lenient_bubbles.append(c)
            
            if len(lenient_bubbles) >= CHOICES_PER_QUESTION:
                bubbles = sorted(lenient_bubbles, key=cv2.contourArea, reverse=True)[:CHOICES_PER_QUESTION]
            else:
                return "Invalid_Layout"
        elif len(bubbles) > CHOICES_PER_QUESTION:
            # Take the 5 largest bubbles
            bubbles = sorted(bubbles, key=cv2.contourArea, reverse=True)[:CHOICES_PER_QUESTION]
        
        # Sort bubbles left to right
        bubbles.sort(key=lambda b: cv2.boundingRect(b)[0])
        
        # Analyze fill levels
        pixel_counts = []
        for bubble_contour in bubbles:
            mask = np.zeros(question_roi_binary.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble_contour], -1, 255, -1)
            masked_roi = cv2.bitwise_and(question_roi_binary, question_roi_binary, mask=mask)
            pixel_count = cv2.countNonZero(masked_roi)
            pixel_counts.append(pixel_count)
        
        if debug:
            print(f"    Pixel counts: {pixel_counts}")
        
        if not pixel_counts:
            return "No_Bubbles"
        
        max_pixels = max(pixel_counts)
        
        if max_pixels < MIN_FILLED_PIXEL_COUNT:
            return "Skipped"
        
        # Check for ambiguity
        sorted_counts = sorted(pixel_counts, reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[1] / sorted_counts[0] > AMBIGUITY_RATIO_THRESH:
            return "Invalid"
        
        marked_index = np.argmax(pixel_counts)
        return self.answer_labels[marked_index]

    def process_sheet(self, image_path, debug=False):
        """
        Main processing function with fixed layout detection for 48 questions.
        """
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"error": f"Could not read image: {image_path}"}

        print(f"Original image shape: {original_image.shape}")

        # Perspective correction
        warped_color = self._get_perspective_transform(original_image)
        
        if debug:
            cv2.imshow("Warped Image", cv2.resize(warped_color, (500, 707)))
            cv2.waitKey(0)
        
        # Binarization
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(warped_gray, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 15, 7)
        
        _, otsu_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binarized = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
        
        kernel = np.ones((2, 2), np.uint8)
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

        if debug:
            cv2.imshow("Binarized Image", cv2.resize(binarized, (500, 707)))
            cv2.waitKey(0)

        # Layout discovery with column-first approach
        question_layout = self._discover_question_layout(binarized)
        
        print(f"Detected {len(question_layout)} questions")
        
        # Validate that we have exactly 48 questions
        if len(question_layout) != TOTAL_EXPECTED_QUESTIONS:
            print(f"Warning: Expected {TOTAL_EXPECTED_QUESTIONS} questions, but detected {len(question_layout)}")
        
        if debug:
            debug_img = warped_color.copy()
            for item in question_layout:
                q_num = item['question_num']
                x, y, w, h = item['roi']
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, str(q_num), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Detected Question Layout", cv2.resize(debug_img, (500, 707)))
            cv2.waitKey(0)

        # Analyze each question
        student_answers = {}
        
        for item in question_layout:
            q_num = item['question_num']
            x, y, w, h = item['roi']
            
            question_roi_binary = binarized[y:y+h, x:x+w]
            answer = self._analyze_bubbles_in_roi(question_roi_binary, debug)
            
            student_answers[q_num] = answer
            
            if debug:
                print(f"Question {q_num}: {answer}")
        
        if debug:
            cv2.destroyAllWindows()
            
        return student_answers

    def save_answers_to_csv(self, answers_dict, output_path):
        """
        Saves answers to CSV file.
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
    Compares student answers with master answer key.
    """
    try:
        student_df = pd.read_csv(student_csv_path)
        master_df = pd.read_csv(master_csv_path)
    except FileNotFoundError as e:
        return {"error": f"File not found: {e.filename}"}
   
    required_cols = ['Question', 'Answer']
    if not all(col in student_df.columns for col in required_cols):
        return {"error": f"Student CSV missing columns. Found: {student_df.columns.tolist()}"}
    if not all(col in master_df.columns for col in required_cols):
        return {"error": f"Master CSV missing columns. Found: {master_df.columns.tolist()}"}
       
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
    Main function with fixed OMR processing for 48 questions.
    """
    image_to_process = 'student_sheet_filled.png'
    student_answers_csv = 'student_answers.csv'
    master_answers_csv = 'master_answers.csv'
    
    grader = FixedOMRGrader()
    
    print(f"Processing '{image_to_process}' with fixed algorithm (48 questions expected)...")
    student_answers = grader.process_sheet(image_to_process, debug=True)
    
    if "error" in student_answers:
        print(f"Error: {student_answers['error']}")
        return
    
    print(f"\nProcessed {len(student_answers)} questions")
    print("\nAnswers found:")
    for q_num, answer in sorted(student_answers.items()):
        print(f"Question {q_num}: {answer}")
    
    if grader.save_answers_to_csv(student_answers, student_answers_csv):
        print(f"\nAnswers saved to '{student_answers_csv}'")
    
    if os.path.exists(master_answers_csv):
        print(f"\nComparing with master answer key...")
        results = grade_from_csv(student_answers_csv, master_answers_csv)
        
        if "error" in results:
            print(f"Grading error: {results['error']}")
        else:
            print(f"\n--- GRADING RESULTS ---")
            print(f"Score: {results['score']} / {results['total_questions']}")
            print(f"Percentage: {(results['score'] / results['total_questions']) * 100:.1f}%")

if __name__ == '__main__':
    main()