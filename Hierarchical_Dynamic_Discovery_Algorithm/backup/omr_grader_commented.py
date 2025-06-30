# simple_omr_grader.py
import cv2
import numpy as np
import pandas as pd
import sys
import os
import scipy.signal
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration Constants ---
DEFAULT_IMAGE = 'student_sheet_filled.png'  # Default image file to process if none provided
RESULTS_CSV = 'student_answers.csv'  # Output CSV file for storing results

# Bubble detection parameters - these control what shapes are considered valid bubbles
BUBBLE_MIN_AREA = 150  # Minimum area (in pixels) for a contour to be considered a bubble
BUBBLE_MAX_AREA = 2500  # Maximum area (in pixels) for a contour to be considered a bubble
BUBBLE_MIN_CIRCULARITY = 0.7  # Minimum circularity score (0-1) - higher means more circular
BUBBLE_ASPECT_RATIO_RANGE = (0.7, 1.3)  # Acceptable width/height ratio range for bubbles


def is_bubble(contour):
    """
    Determines if a contour represents a valid bubble based on geometric properties.
    
    Parameters:
    -----------
    contour : numpy.ndarray
        A contour array from OpenCV containing points that form a closed shape
        Shape: (N, 1, 2) where N is the number of points, each point has (x, y) coordinates
        
    Returns:
    --------
    bool
        True if the contour meets all bubble criteria, False otherwise
        
    Criteria checked:
    - Area within acceptable range (BUBBLE_MIN_AREA to BUBBLE_MAX_AREA)
    - Circularity above threshold (BUBBLE_MIN_CIRCULARITY)
    - Aspect ratio within acceptable range (BUBBLE_ASPECT_RATIO_RANGE)
    
    Possible variations:
    - Adjust area thresholds for different bubble sizes
    - Modify circularity threshold for more/less strict circularity requirements
    - Change aspect ratio range for elliptical vs circular bubbles
    """
    # Calculate the area enclosed by the contour
    # cv2.contourArea() computes the area using Green's theorem
    # For a closed contour, this gives the total enclosed area in square pixels
    area = cv2.contourArea(contour)
    if not (BUBBLE_MIN_AREA < area < BUBBLE_MAX_AREA):
        return False
    
    # Calculate the perimeter (circumference) of the contour
    # cv2.arcLength() computes the perimeter by summing distances between consecutive points
    # True parameter means the contour is closed (last point connects to first point)
    perimeter = cv2.arcLength(contour, True)  # True means contour is closed
    if perimeter == 0:
        return False
    
    # Calculate circularity: 4π * area / (perimeter²)
    # This is a measure of how circular a shape is:
    # - Perfect circle: circularity = 1.0
    # - Less circular shapes: circularity < 1.0
    # - More irregular shapes: circularity approaches 0
    # Formula explanation:
    # - For a perfect circle: area = πr², perimeter = 2πr
    # - So: 4π * area / (perimeter²) = 4π * πr² / (2πr)² = 4π²r² / 4π²r² = 1
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < BUBBLE_MIN_CIRCULARITY:
        return False
    
    # Get bounding rectangle and calculate aspect ratio
    # cv2.boundingRect() returns the smallest rectangle that contains the contour
    # Returns: (x, y, width, height) where (x,y) is top-left corner
    x, y, w, h = cv2.boundingRect(contour)  # Returns (x, y, width, height)
    aspect_ratio = float(w) / h  # width divided by height
    if not (BUBBLE_ASPECT_RATIO_RANGE[0] <= aspect_ratio <= BUBBLE_ASPECT_RATIO_RANGE[1]):
        return False
    
    return True


def is_in_corner(cx, cy, img_shape, margin=100):
    """
    Checks if a point is located in one of the four corners of the image.
    Used to filter out QR codes or other corner elements that might be detected as bubbles.
    
    Parameters:
    -----------
    cx : int
        X-coordinate of the point to check
    cy : int
        Y-coordinate of the point to check
    img_shape : tuple
        Shape of the image (height, width, channels) from image.shape
    margin : int, optional (default=100)
        Distance from the edge to consider as "corner region"
        
    Returns:
    --------
    bool
        True if point is in any corner region, False otherwise
        
    Corner regions defined:
    - Top-left: (0, 0) to (margin, margin)
    - Top-right: (width-margin, 0) to (width, margin)
    - Bottom-left: (0, height-margin) to (margin, height)
    - Bottom-right: (width-margin, height-margin) to (width, height)
    """
    h, w = img_shape[:2]  # Extract height and width from image shape
    return (
        (cx < margin and cy < margin) or  # top-left corner
        (cx > w - margin and cy < margin) or  # top-right corner
        (cx < margin and cy > h - margin) or  # bottom-left corner
        (cx > w - margin and cy > h - margin)   # bottom-right corner
    )


# def kmeans_1d(xs, k=3, n_iter=20):
#     """
#     Performs 1-dimensional K-means clustering on a set of values.
#     Used for grouping similar values together (e.g., grouping bubble coordinates).
    
#     Parameters:
#     -----------
#     xs : numpy.ndarray
#         1D array of values to cluster
#     k : int, optional (default=3)
#         Number of clusters to create
#     n_iter : int, optional (default=20)
#         Maximum number of iterations for the clustering algorithm
        
#     Returns:
#     --------
#     labels : numpy.ndarray
#         Array of cluster labels (0 to k-1) for each input value
#         Shape: same as xs
#     centers : numpy.ndarray
#         Array of cluster center values
#         Shape: (k,)
        
#     Algorithm:
#     1. Initialize centers evenly spaced across the data range
#     2. Assign each point to nearest center
#     3. Update centers as mean of assigned points
#     4. Repeat until convergence or max iterations
#     """
#     # Initialize centers evenly across the data range
#     # np.linspace creates k evenly spaced values from min to max
#     centers = np.linspace(xs.min(), xs.max(), k)
    
#     for _ in range(n_iter):
#         # Assign each point to the nearest center
#         # xs[:, None] creates a column vector, centers[None, :] creates a row vector
#         # This creates a distance matrix where each row is distances from one point to all centers
#         # Broadcasting explanation:
#         # - xs[:, None] has shape (n, 1) - each point as a column
#         # - centers[None, :] has shape (1, k) - each center as a row
#         # - xs[:, None] - centers[None, :] creates (n, k) matrix of differences
#         # - np.abs() takes absolute value of differences
#         # - np.argmin(axis=1) finds index of minimum value in each row (closest center)
#         labels = np.argmin(np.abs(xs[:, None] - centers[None, :]), axis=1)
        
#         # Update centers as the mean of all points assigned to each cluster
#         for i in range(k):
#             if np.any(labels == i):  # Check if any points were assigned to this cluster
#                 centers[i] = xs[labels == i].mean()  # Calculate mean of assigned points
    
#     return labels, centers


def main():
    """
    Main function that orchestrates the entire OMR sheet processing workflow.
    
    Workflow:
    1. Load and validate input image
    2. Preprocess image (grayscale, binarization)
    3. Detect bubble contours
    4. Filter and validate bubbles
    5. Group bubbles into rows and questions
    6. Extract student answers
    7. Compare with master answers
    8. Generate results and visualizations
    
    Command line usage:
    python omr_grader.py [image_path]
    
    If no image path provided, uses DEFAULT_IMAGE constant
    """
    # Parse command line arguments
    if len(sys.argv) < 2:
        image_path = DEFAULT_IMAGE
        print(f"Usage: python {sys.argv[0]} <path_to_image>\nUsing default '{image_path}'")
    else:
        image_path = sys.argv[1]  # Get image path from command line argument

    # 1. Load image
    image = cv2.imread(image_path)  # Load image in BGR format (OpenCV default)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return
    print(f"Loaded image: {image_path} shape={image.shape}")  # shape is (height, width, channels)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    print(f"Converted to grayscale. Shape: {gray.shape}")  # shape is (height, width)

    # 3. Binarize (Otsu's threshold)
    # Otsu's method automatically finds optimal threshold by maximizing inter-class variance
    # THRESH_BINARY_INV: White pixels (255) for values above threshold, black (0) below
    # THRESH_OTSU: Automatically determine optimal threshold
    # The underscore (_) is the threshold value returned by cv2.threshold (we don't use it)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Binarized image using Otsu's threshold.")

    # 4. Detect bubbles (contours)
    # Morphological closing to fill small gaps in bubble boundaries
    # Create a 3x3 kernel for morphological operations
    # A 3x3 kernel is a small sliding window that will be applied to each pixel in the image
    # The kernel is a binary (0 or 255) image the same size as the image
    # We use a 3x3 kernel here to fill small gaps in bubble boundaries
    # The kernel is a square because we want to apply the same operation in both x and y directions
    kernel = np.ones((3, 3), np.uint8)  # Create a 3x3 kernel with all pixels set to 255 (white)

    # Apply morphological closing to the binarized image
    # MORPH_CLOSE: dilation followed by erosion
    # Dilation: expand bright regions (white pixels) by adding a pixel to each boundary
    # Erosion: shrink bright regions (white pixels) by removing a pixel from each boundary
    # Closing: dilation followed by erosion
    # This fills small holes and connects nearby components in the image
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    # The result is a binarized image with bubbles connected and holes filled in
    # MORPH_CLOSE: dilation followed by erosion - fills small holes and connects nearby components
    
    # Find contours using RETR_CCOMP (two-level hierarchy)
    # RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy
    # - External contours (parent = -1)
    # Find contours using RETR_CCOMP (two-level hierarchy)
    # RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy
    # - External contours (parent = -1 in hierarchy)
    # Find contours in the binarized image using RETR_CCOMP retrieval mode and CHAIN_APPROX_SIMPLE approximation method
    # cv2.RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy
    # - External contours (parent = -1 in hierarchy)
    # - Internal contours (holes, parent = index of external contour)
    # cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments, leaving only their end points
    # This method reduces the number of points stored, resulting in a more efficient contour representation
    # Find contours using RETR_CCOMP (two-level hierarchy)
    # RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy
    # - External contours (parent = -1 in hierarchy)
    # - Internal contours (holes, parent = index of external contour)
    # cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments, leaving only their end points
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Explain like I'm 5 (ELI5):
    # Imagine you have a bunch of loops drawn on a piece of paper, some loops inside others.
    # The hierarchy is like a list that tells you which loops are outside (big loops) and which are inside (small loops inside big ones).
    # This helps us know the relationship between the loops, like which ones are holes inside others.
    
    print(f"Total contours found: {len(contours)}")  # Prints the number of contours found

    bubbles = []
    # Process only external contours (parent index = -1 in hierarchy)
    if hierarchy is not None:
        for i, c in enumerate(contours):
            # hierarchy[0][i][3] is the parent index for contour i 
            # [3] means the parent contour index in the hierarchy (parent contour) 
            # -1 means it's an external contour (no parent)
            if hierarchy[0][i][3] == -1:  # Check if it's an external contour
                if is_bubble(c):
                    # Calculate centroid (center of mass) of the contour
                    # cv2.moments() returns a dictionary of moment values
                    # Moments are mathematical descriptors of the shape's properties
                    M = cv2.moments(c)  # Returns dictionary of moment values
                    if M['m00'] != 0:  # m00 is the area, check it's not zero
                        # CENTROID FORMULA EXPLANATION:
                        # The centroid (center of mass) of a shape is calculated using moments:
                        # cx = M10 / M00  (x-coordinate of centroid)
                        # cy = M01 / M00  (y-coordinate of centroid)
                        # Where:
                        # - M00 = area of the shape (sum of all pixels)
                        # - M10 = first moment about x-axis (sum of x-coordinates of all pixels)
                        # - M01 = first moment about y-axis (sum of y-coordinates of all pixels)
                        # - M11 = mixed second moment about x and y axes (sum of products of x and y coordinates of all pixels)
                        # 
                        # Mathematical derivation:
                        # For a discrete shape with pixels at positions (xi, yi):
                        # M00 = Σ(1) = total number of pixels = area
                        # M10 = Σ(xi) = sum of all x-coordinates
                        # M01 = Σ(yi) = sum of all y-coordinates
                        # 
                        # The centroid is the "average position" of all pixels:
                        # cx = (Σ(xi)) / (Σ(1)) = M10 / M00
                        # cy = (Σ(yi)) / (Σ(1)) = M01 / M00
                        
                        # Example: a shape with 4 pixels at (0,0), (1,0), (2,0), (3,0)
                        # M00 = 4 (total number of pixels)
                        # M10 = 6 (sum of x-coordinates: 0+1+2+3)
                        # M01 = 0 (sum of y-coordinates: all 0)
                        # 
                        # The centroid is the "average position" of all pixels:
                        # cx = 6/4 = 1.5 (average x-coordinate)
                        # cy = 0/4 = 0 (average y-coordinate)
                        
                        cx = int(M['m10'] / M['m00'])  # x-coordinate of centroid
                        cy = int(M['m01'] / M['m00'])  # y-coordinate of centroid
                        if is_in_corner(cx, cy, image.shape, margin=100):
                            continue  # skip QR code regions
                        bubbles.append({'cx': cx, 'cy': cy, 'contour': c})
    print(f"Potential bubbles detected: {len(bubbles)}")

    # Print all detected bubble x-coordinates for debugging
    xs = np.array([b['cx'] for b in bubbles])  # Extract x-coordinates from all bubbles
    print('All detected bubble x-coordinates:', np.sort(xs))

    # 5. Group all bubbles into 16 rows by y-coordinate (structure-based, no columns)
    ys = np.array([b['cy'] for b in bubbles])  # Extract y-coordinates from all bubbles
    # Create 17 percentile points (0%, 6.25%, 12.5%, ..., 100%) to define 16 rows
    # np.linspace(0, 100, 17) creates: [0, 6.25, 12.5, 18.75, 25, 31.25, 37.5, 43.75, 50, 56.25, 62.5, 68.75, 75, 81.25, 87.5, 93.75, 100]
    # np.percentile(ys, ...) finds the y-coordinate values at these percentile points
    row_edges = np.percentile(ys, np.linspace(0, 100, 17))
    for b in bubbles:
        for r in range(16):
            # Check if bubble's y-coordinate falls within this row's range
            # row_edges[r] is the lower boundary of row r
            # row_edges[r+1] is the upper boundary of row r
            if row_edges[r] <= b['cy'] <= row_edges[r+1]:
                b['row'] = r  # Assign row number (0-15)
                break

    # 6. For each row, sort by x and assign to questions/choices
    question_bubbles = []
    for r in range(16):  # Process each of the 16 rows
        
        # Filter the bubbles list to get only those bubbles that belong to the current row `r`
        # - `bubbles` is a list of dictionaries, where each dictionary represents a bubble
        # - Each dictionary contains information such as 'cx', 'cy', 'contour', and 'row'
        # - The goal is to extract bubbles that have their 'row' key set to the current row number `r`
        # - This is achieved using a list comprehension that iterates over `bubbles`
        # - `b.get('row')` retrieves the value associated with the 'row' key for each bubble `b`
        # - `b.get('row') == r` checks if the bubble belongs to the current row `r`
        # - Only bubbles satisfying this condition are included in the resulting `row_bubbles` list
        row_bubbles = [b for b in bubbles if b.get('row') == r]  # Get bubbles in this row
        if len(row_bubbles) != 15:
            print(f"Row {r+1}: {len(row_bubbles)} bubbles detected (expected 15)")
            continue
        # Sort the bubbles in this row by their x-coordinate (left to right)
        # 
        # EXPLANATION:
        # - `sorted()` is a built-in Python function that takes a list and returns a new sorted list
        # - We pass `row_bubbles` as the list to sort
        # - `key=lambda b: b['cx']` specifies how to sort the list
        # - `lambda b: b['cx']` is an anonymous function that takes a bubble `b` and returns its 'cx' value
        # - `reverse=True` tells `sorted()` to sort the list in reverse (descending) order by the 'cx' values of the bubbles
        # 
        # The sorted list is stored in `row_bubbles_sorted`
        row_bubbles_sorted = sorted(row_bubbles, key=lambda b: b['cx'])
        for i, b in enumerate(row_bubbles_sorted):
            # Calculate question number: each row has 3 questions with 5 choices each
            # Formula: row_number + 1 + (bubble_index // 5) * 16
            # This creates question numbers: 1, 17, 33 for row 0; 2, 18, 34 for row 1; etc.
            # 
            # EXPLANATION OF THE FORMULA:
            # - row_number + 1: Base question number for this row (1-16)
            # - bubble_index // 5: Which question block in this row (0, 1, or 2)
            # - * 16: Skip 16 questions to get to the next block
            # 
            # Example for row 0:
            # - Bubble 0-4: i//5 = 0, so q_num = 0+1+0*16 = 1
            # - Bubble 5-9: i//5 = 1, so q_num = 0+1+1*16 = 17  
            # - Bubble 10-14: i//5 = 2, so q_num = 0+1+2*16 = 33
            q_num = r + 1 + (i // 5) * 16
            b['question'] = q_num
            b['choice'] = i % 5  # 0=A, 1=B, 2=C, 3=D, 4=E (modulo gives remainder)
            question_bubbles.append(b)

    # 7. Extract answers for each question
    # Group bubbles by question number
    # Example of using a defaultdict:
    # - Create a dictionary where each key has a default value (in this case, an empty list)
    # - When you access a key that doesn't exist, it is automatically created with the default value
    # - When you access a key that does exist, its value is returned as usual
    
    # Each question has 5 bubbles, so use a list of lists to store them
    # question_to_bubbles will be a dictionary with 48 keys (questions)
    # Each key's value will be a list of 5 bubbles (A-E)
    
    question_to_bubbles = defaultdict(list)  # Dictionary with default empty list
    for b in question_bubbles:
        question_to_bubbles[b['question']].append(b)  # Add bubble to its question's list

    answers = []
    for q in range(1, 49):  # Process all 48 questions
        bubbles = question_to_bubbles.get(q, [])  # Get bubbles for this question
        if len(bubbles) != 5:
            print(f"Question {q}: {len(bubbles)} bubbles found (expected 5)")
            answers.append({'question': q, 'answer': ''})
            continue
        
        # For each bubble, compute mean pixel value inside the contour (on binarized image)
        fill_scores = []
        for b in bubbles:
            # Create a mask for this bubble's contour
            # Create a mask image for this bubble's contour
            # A mask image is a binary image (0 or 255) where:
            # - 0 represents pixels outside the contour
            # - 255 represents pixels inside the contour
            #
            # We use this mask to calculate the mean pixel value inside the bubble
            # The mask is a single-channel image (grayscale) with same height and width as the binarized image
            mask = np.zeros(binarized.shape, dtype=np.uint8)  # Create black background (all zeros)
            
            # Draw the bubble's contour on the mask image
            # -1 as the contour index means draw all contours in the list (there's only one)
            # 255 as the color means fill the contour with white (255)
            # -1 as the thickness means fill the contour completely (i.e., no border)
            cv2.drawContours(mask, [b['contour']], -1, 255, -1)  # Fill contour with white
            # Calculate mean pixel value inside the bubble
            # Higher values mean more filled (since image is inverted)
            # 
            # FILL SCORE CALCULATION EXPLANATION:
            # - binarized image: 0 = black (empty), 255 = white (filled) - because we used THRESH_BINARY_INV
            # - mask: 0 = outside bubble (black), 255 = inside bubble (white)
            # 
            # The mask is used to select only the pixels inside the bubble
            # for calculating the mean pixel value. The mask is a binary image
            # where white (255) represents pixels inside the bubble, and black (0)
            # represents pixels outside the bubble.
            # 
            # When we pass this mask to cv2.mean(), it will only consider the pixels
            # where the mask is white (255) for calculating the mean pixel value.
            # This is equivalent to selecting only the pixels inside the bubble
            # and calculating the average of those pixels.
            
            # - cv2.mean() calculates: sum(pixel_values * mask_values) / sum(mask_values)
            # - Since mask is 255 inside bubble and 0 outside, this gives the average pixel value inside the bubble
            # - Result is 0-255: higher means more white pixels (more filled) inside the bubble
            # Calculate mean pixel value inside the bubble
            # cv2.mean() takes the following arguments:
            # - image: the binarized image (grayscale)
            # - mask: the mask image we created earlier, where:
            #   - 0 represents pixels outside the bubble (black)
            #   - 255 represents pixels inside the bubble (white)
            # - computation: the mean pixel value calculation
            #   - sum(pixel_values * mask_values) / sum(mask_values)
            #   - Since mask is 255 inside bubble and 0 outside, this gives the average pixel value inside the bubble
            #   - Result is 0-255: higher means more white pixels (more filled) inside the bubble
            
            # cv2.mean() returns a tuple of 3 values: (mean of B channel, mean of G channel, mean of R channel)
            # For grayscale images, all 3 channels have the same value, so we can use any of them
            # The [0] indexing gets the first (and only) channel, which is the grayscale value
            mean_val = cv2.mean(binarized, mask=mask)[0]  # [0] gets the first channel (grayscale)
            fill_scores.append(mean_val)
        
        # The filled bubble should have the highest mean (most white in inverted image)
        filled_idx = np.argmax(fill_scores)  # Index of bubble with highest fill score
        
        # Heuristic: Only accept as filled if it's significantly higher than the next
        # This prevents false positives from slight variations
        sorted_scores = sorted(fill_scores, reverse=True)  # Sort scores in descending order
        if sorted_scores[0] - sorted_scores[1] < 20:  # threshold can be tuned
            answer = ''  # ambiguous - no clear winner
        else:
            # Convert index to letter: 0->A, 1->B, 2->C, 3->D, 4->E
            # chr(ord('A') + filled_idx) explanation:
            # - ord('A') = 65 (ASCII value of 'A')
            # - ord('A') + 0 = 65 -> chr(65) = 'A'
            # - ord('A') + 1 = 66 -> chr(66) = 'B'
            # - ord('A') + 2 = 67 -> chr(67) = 'C'
            # - etc.
            answer = chr(ord('A') + filled_idx)
        answers.append({'question': q, 'answer': answer})

    # 8. Save answers to CSV
    answers_df = pd.DataFrame(answers)  # Convert list of dictionaries to DataFrame
    answers_df.to_csv('student_answers.csv', index=False)  # Save without row indices
    print("Saved student answers to student_answers.csv")

    # 8b. Compare with master answers and write results with marks
    master_df = pd.read_csv('master_answers.csv')  # Load correct answers
    # Ensure both dataframes are sorted by question for proper alignment
    answers_df = answers_df.sort_values('question').reset_index(drop=True)
    master_df = master_df.sort_values('Question').reset_index(drop=True)
    
    results = []
    total = len(master_df)  # Total number of questions
    obtained = 0  # Number of correct answers
    
    for i in range(total):
        # Extract data from DataFrames using .loc[row_index, column_name]
        qnum = master_df.loc[i, 'Question']  # Question number from master sheet
        correct = master_df.loc[i, 'Answer']  # Correct answer from master sheet
        student = answers_df.loc[i, 'answer'] if i < len(answers_df) else ''  # Student's answer
        
        # COMPARISON LOGIC EXPLANATION:
        # - str(student).strip().upper(): Convert to string, remove whitespace, convert to uppercase
        # - str(correct).strip().upper(): Same for correct answer
        # - == : Compare the normalized strings
        # - int(...): Convert boolean result to integer (True=1, False=0)
        # This handles cases like: 'a' == 'A', ' A ' == 'A', etc.
        mark = int(str(student).strip().upper() == str(correct).strip().upper())
        obtained += mark
        results.append({'question number': qnum, 'correct answer': correct, 'student answer': student, 'marks': mark})
    
    # Calculate percentage score
    # Formula: (obtained_marks / total_questions) * 100
    # The if total else 0.0 prevents division by zero
    percent = 100.0 * obtained / total if total else 0.0
    
    # Add summary rows to results
    results.append({'question number': 'Total', 'correct answer': total, 'student answer': 'Obtained', 'marks': obtained})
    results.append({'question number': 'Percentage', 'correct answer': '', 'student answer': '', 'marks': f'{percent:.2f}%'})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('student_answers.csv', index=False)
    print(f"Results written to student_answers.csv. Score: {obtained}/{total} ({percent:.2f}%)")

    # 9. Visualize bubbles by row (no columns)
    vis = image.copy()  # Create a copy for visualization
    
    # Define colors for different elements (BGR format - OpenCV uses Blue, Green, Red)
    row_color = (255, 0, 0)  # Blue for row highlight (BGR format)
    correct_color = (0, 255, 0)  # Green for correct answers
    wrong_color = (0, 0, 255)    # Red for wrong answers
    bubble_color = (255, 0, 0)   # Blue for all bubbles
    text_color = (0, 0, 0)       # Black for text
    
    # Define drawing parameters
    row_thickness = 2
    bubble_thickness = 2
    alpha = 0.15  # Transparency for row highlight (0=transparent, 1=opaque)

    # Prepare master answers for quick lookup
    # Convert DataFrame to dictionary for faster access
    # Dictionary format: {question_number: answer}
    master_answers = {int(row['Question']): row['Answer'] for _, row in master_df.iterrows()}
    student_answers = {int(row['question']): row['answer'] for _, row in answers_df.iterrows()}

    # Draw row highlights and bubbles
    for r in range(16):  # Process each of the 16 rows
        row_bubbles = [b for b in question_bubbles if b.get('row') == r]  # Get bubbles in this row
        if not row_bubbles:
            continue
        
        # Calculate bounding box for this row
        # min() and max() with generator expressions find the extreme coordinates
        min_y = min(b['cy'] for b in row_bubbles)  # Minimum y-coordinate in row
        max_y = max(b['cy'] for b in row_bubbles)  # Maximum y-coordinate in row
        min_x = min(b['cx'] for b in row_bubbles)  # Minimum x-coordinate in row
        max_x = max(b['cx'] for b in row_bubbles)  # Maximum x-coordinate in row
        
        # Draw semi-transparent blue rectangle highlighting the row
        overlay = vis.copy()
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # pt1 = (min_x-25, min_y-25): top-left corner with 25px margin
        # pt2 = (max_x+25, max_y+25): bottom-right corner with 25px margin
        # thickness = -1: filled rectangle
        cv2.rectangle(overlay, (min_x-25, min_y-25), (max_x+25, max_y+25), row_color, -1)
        # cv2.addWeighted blends two images: overlay * alpha + vis * (1-alpha)
        
        # Blend the semi-transparent blue rectangle (overlay) with the original image (vis)
        # - cv2.addWeighted() takes two images, a weight (alpha) for the first, and a weight (1-alpha) for the second
        # - The result is a new image with the two images blended by the weights
        # - The last argument (0) is the gamma value (not used in this case)
        # - The result is stored back in the original image (vis)
        # This effectively draws the blue row highlight on top of the original image
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        
        # Draw all bubbles in blue first
        for b in row_bubbles:
            cv2.drawContours(vis, [b['contour']], -1, bubble_color, bubble_thickness)
            # cv2.drawContours(
                # image,         # The image to draw on
                # contours,      # List of contour(s) to draw
                # contourIndex,  # Index of contour to draw (-1 = all)
                # color,         # Color to draw (BGR format)
                # thickness      # Line thickness
                    # )


    # Draw selected answers (green/red) and overlay question numbers
    for q in range(1, 49):  # Process all 48 questions
        bubbles = question_to_bubbles.get(q, [])  # Get bubbles for this question
        if len(bubbles) != 5:
            continue
        
        # Find which bubble is selected by student
        student_ans = student_answers.get(q, '')  # Student's answer for this question
        correct_ans = master_answers.get(q, '')   # Correct answer for this question
        
        for idx, b in enumerate(bubbles):
            # Convert bubble index to choice letter: 0->A, 1->B, 2->C, 3->D, 4->E
            choice_letter = chr(ord('A') + idx)
            if student_ans == choice_letter:
                # Selected answer: green if correct, red if wrong
                color = correct_color if student_ans == correct_ans else wrong_color
                # Draw thicker contour (thickness=4) for selected answers
                cv2.drawContours(vis, [b['contour']], -1, color, 4)
            
            # Overlay question number on the first bubble of each question
            if idx == 0:
                q_text = str(q)  # Question number as string
                # Get text size for centering
                # cv2.getTextSize(text, font, scale, thickness) returns (text_size, baseline)
                # Get the size of the question number text for centering.
                # cv2.getTextSize() returns a tuple of (text_size, baseline) where:
                # - text_size is a tuple of (width, height) of the text in pixels
                # - baseline is the y-coordinate of the baseline relative to the bottom of the text
                # We only care about the width here, so we use _ to ignore the baseline value.
                # The arguments to cv2.getTextSize() are:
                #   text: the string text to measure
                #   fontFace: the font face to use (cv2.FONT_HERSHEY_SIMPLEX in this case)
                #   fontScale: the scale factor to apply to the font (0.5 here)
                #   thickness: the line thickness to use when drawing the text (1 here)
                
                (text_width, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw question number above the bubble
                # Position: (bubble_center_x - text_width/2, bubble_center_y - 10)
                # This centers the text horizontally and places it 10 pixels above the bubble
                # Draw question number above the bubble
                # This code positions the question number in the center of the bubble,
                # 10 pixels above the bubble.
                # The arguments to cv2.putText() are:
                #   image: the image to draw the text on (vis)
                #   text: the string text to draw (q_text)
                #   org: the origin (top-left corner) of the text (tuple of (x, y))
                #   fontFace: the font face to use (cv2.FONT_HERSHEY_SIMPLEX in this case)
                #   fontScale: the scale factor to apply to the font (0.5 here)
                #   color: the color of the text (text_color)
                #   thickness: the line thickness to use when drawing the text (2 here)
                # Calculate the x-coordinate of the text origin:
                #   bubble_center_x - text_width/2
                # This centers the text horizontally.
                # Use the bubble's y-coordinate minus 10 as the y-coordinate of the text origin.
                # This positions the text 10 pixels above the bubble.
                cv2.putText(vis, q_text, (b['cx'] - text_width // 2, b['cy'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Save the output image
    output_filename = "omr_graded_output.png"
    cv2.imwrite(output_filename, vis)
    print(f"Saved graded image to {output_filename}")

    # Display images for user inspection
    cv2.imshow("Graded Sheet", vis)  # Show the annotated image
    cv2.imshow("Binarized", binarized)  # Show the binary image for debugging
    print("Press any key in the image window to exit...")
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close all windows

    # 10. Load results CSV (if exists)
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)  # Load the results file
        print(f"Loaded results CSV: {RESULTS_CSV}")
        print(df.head())  # Show first few rows
    else:
        print(f"Results CSV not found: {RESULTS_CSV}")


if __name__ == '__main__':
    main()  # Execute main function when script is run directly
