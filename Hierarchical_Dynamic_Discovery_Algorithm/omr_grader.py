# simple_omr_grader.py
import cv2
import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict

# --- Configuration Constants ---
DEFAULT_IMAGE = 'student_sheet_filled.png'  # Default image file to process if none provided
RESULTS_CSV = 'student_answers.csv'  # Output CSV file for storing results

# Bubble detection parameters
BUBBLE_MIN_AREA = 150
BUBBLE_MAX_AREA = 2500
BUBBLE_MIN_CIRCULARITY = 0.7
BUBBLE_ASPECT_RATIO_RANGE = (0.7, 1.3)


def is_bubble(contour):
    """
    Determines if a contour represents a valid bubble based on geometric properties.
    """
    area = cv2.contourArea(contour)
    if not (BUBBLE_MIN_AREA < area < BUBBLE_MAX_AREA):
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < BUBBLE_MIN_CIRCULARITY:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if not (BUBBLE_ASPECT_RATIO_RANGE[0] <= aspect_ratio <= BUBBLE_ASPECT_RATIO_RANGE[1]):
        return False
    return True


def is_in_corner(cx, cy, img_shape, margin=100):
    """
    Checks if a point is located in one of the four corners of the image.
    """
    h, w = img_shape[:2]
    return (
        (cx < margin and cy < margin) or
        (cx > w - margin and cy < margin) or
        (cx < margin and cy > h - margin) or
        (cx > w - margin and cy > h - margin)
    )

# def kmeans_1d(xs, k=3, n_iter=20):
#     centers = np.linspace(xs.min(), xs.max(), k)
    
#     for _ in range(n_iter):
#         labels = np.argmin(np.abs(xs[:, None] - centers[None, :]), axis=1)
#         for i in range(k):
#             if np.any(labels == i):  # Check if any points were assigned to this cluster
#                 centers[i] = xs[labels == i].mean()  # Calculate mean of assigned points
    
#     return labels, centers

def main():
    """
    Main function that orchestrates the entire OMR sheet processing workflow.
    """
    # Parse command line arguments
    if len(sys.argv) < 2:
        image_path = DEFAULT_IMAGE
        print(f"Usage: python {sys.argv[0]} <path_to_image>\nUsing default '{image_path}'")
    else:
        image_path = sys.argv[1]

    # 1. Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return
    print(f"Loaded image: {image_path} shape={image.shape}")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Converted to grayscale. Shape: {gray.shape}")

    # 3. Binarize (Otsu's threshold)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Binarized image using Otsu's threshold.")

    # 4. Detect bubbles (contours)
    kernel = np.ones((3, 3), np.uint8)
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    bubbles = []
    if hierarchy is not None:
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                if is_bubble(c):
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        if is_in_corner(cx, cy, image.shape, margin=100):
                            continue
                        bubbles.append({'cx': cx, 'cy': cy, 'contour': c})
    print(f"Potential bubbles detected: {len(bubbles)}")

    xs = np.array([b['cx'] for b in bubbles])
    print('All detected bubble x-coordinates:', np.sort(xs))

    # 5. Group all bubbles into 16 rows by y-coordinate
    ys = np.array([b['cy'] for b in bubbles])
    row_edges = np.percentile(ys, np.linspace(0, 100, 17))
    for b in bubbles:
        for r in range(16):
            if row_edges[r] <= b['cy'] <= row_edges[r+1]:
                b['row'] = r
                break

    # 6. For each row, sort by x and assign to questions/choices
    question_bubbles = []
    for r in range(16):
        row_bubbles = [b for b in bubbles if b.get('row') == r]
        if len(row_bubbles) != 15:
            print(f"Row {r+1}: {len(row_bubbles)} bubbles detected (expected 15)")
            continue
        row_bubbles_sorted = sorted(row_bubbles, key=lambda b: b['cx'])
        for i, b in enumerate(row_bubbles_sorted):
            q_num = r + 1 + (i // 5) * 16
            b['question'] = q_num
            b['choice'] = i % 5
            question_bubbles.append(b)

    # 7. Extract answers for each question
    question_to_bubbles = defaultdict(list)
    for b in question_bubbles:
        question_to_bubbles[b['question']].append(b)

    answers = []
    for q in range(1, 49):
        bubbles = question_to_bubbles.get(q, [])
        if len(bubbles) != 5:
            print(f"Question {q}: {len(bubbles)} bubbles found (expected 5)")
            answers.append({'question': q, 'answer': ''})
            continue
        fill_scores = []
        for b in bubbles:
            mask = np.zeros(binarized.shape, dtype=np.uint8)
            cv2.drawContours(mask, [b['contour']], -1, 255, -1)
            mean_val = cv2.mean(binarized, mask=mask)[0]
            fill_scores.append(mean_val)
        filled_idx = np.argmax(fill_scores)
        sorted_scores = sorted(fill_scores, reverse=True)
        if sorted_scores[0] - sorted_scores[1] < 20:
            answer = ''
        else:
            answer = chr(ord('A') + filled_idx)
        answers.append({'question': q, 'answer': answer})

    # 8. Save answers to CSV
    answers_df = pd.DataFrame(answers)
    answers_df.to_csv('student_answers.csv', index=False)
    print("Saved student answers to student_answers.csv")

    # 8b. Compare with master answers and write results with marks
    master_df = pd.read_csv('master_answers.csv')
    answers_df = answers_df.sort_values('question').reset_index(drop=True)
    master_df = master_df.sort_values('Question').reset_index(drop=True)
    results = []
    total = len(master_df)
    obtained = 0
    for i in range(total):
        qnum = master_df.loc[i, 'Question']
        correct = master_df.loc[i, 'Answer']
        student = answers_df.loc[i, 'answer'] if i < len(answers_df) else ''
        mark = int(str(student).strip().upper() == str(correct).strip().upper())
        obtained += mark
        results.append({'question number': qnum, 'correct answer': correct, 'student answer': student, 'marks': mark})
    percent = 100.0 * obtained / total if total else 0.0
    results.append({'question number': 'Total', 'correct answer': total, 'student answer': 'Obtained', 'marks': obtained})
    results.append({'question number': 'Percentage', 'correct answer': '', 'student answer': '', 'marks': f'{percent:.2f}%'})
    results_df = pd.DataFrame(results)
    results_df.to_csv('student_answers.csv', index=False)
    print(f"Results written to student_answers.csv. Score: {obtained}/{total} ({percent:.2f}%)")

    # 9. Visualize bubbles by row
    vis = image.copy()
    row_color = (255, 0, 0)
    correct_color = (0, 255, 0)
    wrong_color = (0, 0, 255)
    bubble_color = (255, 0, 0)
    text_color = (0, 0, 0)
    row_thickness = 2
    bubble_thickness = 2
    alpha = 0.15
    master_answers = {int(row['Question']): row['Answer'] for _, row in master_df.iterrows()}
    student_answers = {int(row['question']): row['answer'] for _, row in answers_df.iterrows()}
    for r in range(16):
        row_bubbles = [b for b in question_bubbles if b.get('row') == r]
        if not row_bubbles:
            continue
        min_y = min(b['cy'] for b in row_bubbles)
        max_y = max(b['cy'] for b in row_bubbles)
        min_x = min(b['cx'] for b in row_bubbles)
        max_x = max(b['cx'] for b in row_bubbles)
        overlay = vis.copy()
        cv2.rectangle(overlay, (min_x-25, min_y-25), (max_x+25, max_y+25), row_color, -1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        for b in row_bubbles:
            cv2.drawContours(vis, [b['contour']], -1, bubble_color, bubble_thickness)

    for q in range(1, 49):
        bubbles = question_to_bubbles.get(q, [])
        if len(bubbles) != 5:
            continue
        student_ans = student_answers.get(q, '')
        correct_ans = master_answers.get(q, '')
        for idx, b in enumerate(bubbles):
            choice_letter = chr(ord('A') + idx)
            if student_ans == choice_letter:
                color = correct_color if student_ans == correct_ans else wrong_color
                cv2.drawContours(vis, [b['contour']], -1, color, 4)
            if idx == 0:
                q_text = str(q)
                (text_width, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(vis, q_text, (b['cx'] - text_width // 2, b['cy'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    output_filename = "omr_graded_output.png"
    cv2.imwrite(output_filename, vis)
    print(f"Saved graded image to {output_filename}")

    cv2.imshow("Graded Sheet", vis)
    cv2.imshow("Binarized", binarized)
    print("Press any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 10. Load results CSV (if exists)
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        print(f"Loaded results CSV: {RESULTS_CSV}")
        print(df.head())
    else:
        print(f"Results CSV not found: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
