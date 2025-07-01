# /omr_batch_processor/src/reporting.py
"""
Functions for generating final reports (CSV and visual image).
"""
import os
import glob
import cv2
import pandas as pd
from . import config

# FUNCTION 1: Saves the individual result for one student
def save_results_csv(results, summary, output_path):
    """Saves the detailed grading results and summary to a specific CSV file."""
    df = pd.DataFrame(results)
    
    summary_rows = pd.DataFrame([
        {'question_number': 'Total', 'correct_answer': summary['total'], 'student_answer': 'Obtained', 'marks': summary['obtained']},
        {'question_number': 'Percentage', 'correct_answer': '', 'student_answer': '', 'marks': f"{summary['percentage']:.2f}%"}
    ])
    df = pd.concat([df, summary_rows], ignore_index=True)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Results successfully saved to {output_path}")
    except IOError as e:
        print(f"Error: Could not write individual result file to {output_path}. Error: {e}")

# FUNCTION 2: Creates the graded image with name and score
def create_visual_feedback(base_image, q_bubbles, student_answers, master_answers, student_name, summary):
    """
    Draws feedback onto the original image, including header info, highlights, and answers.
    """
    vis_image = base_image.copy()
    student_answers_dict = {item['question']: item['answer'] for item in student_answers}

    # Group bubbles by row for processing
    bubbles_by_row = {}
    for b in q_bubbles:
        row_idx = b.get('row')
        if row_idx is not None:
            if row_idx not in bubbles_by_row:
                bubbles_by_row[row_idx] = []
            bubbles_by_row[row_idx].append(b)

    # Draw Name and Score at the top of the sheet
    if 0 in bubbles_by_row:
        top_row_bubbles = bubbles_by_row[0]
        try:
            min_cx_col1 = min(b['cx'] for b in top_row_bubbles if b['question'] in range(1, 17))
            min_cx_col2 = min(b['cx'] for b in top_row_bubbles if b['question'] in range(17, 33))
            y_anchor = min(b['cy'] for b in top_row_bubbles)
            
            name_text = f"Name: {student_name}"
            score_text = f"Score: {summary['obtained']} / {summary['total']} ({summary['percentage']:.1f}%)"
            # Use the specific Y offset for the name
            name_pos = (min_cx_col1 + config.VIS_NAME_X_OFFSET, y_anchor + config.VIS_NAME_Y_OFFSET)
            # Use the specific Y offset for the score
            score_pos = (min_cx_col2 + config.VIS_SCORE_X_OFFSET, y_anchor + config.VIS_SCORE_Y_OFFSET)

            cv2.putText(vis_image, name_text, name_pos, config.VIS_INFO_FONT, config.VIS_INFO_FONT_SCALE, config.VIS_TEXT_COLOR, config.VIS_INFO_FONT_THICKNESS)
            cv2.putText(vis_image, score_text, score_pos, config.VIS_INFO_FONT, config.VIS_INFO_FONT_SCALE, config.VIS_TEXT_COLOR, config.VIS_INFO_FONT_THICKNESS)
        except ValueError:
            print("Warning: Could not find anchor points to draw header text. Skipping.")

    # Draw row highlights
    for row_idx, bubbles in bubbles_by_row.items():
        min_y, max_y = min(b['cy'] for b in bubbles) - 25, max(b['cy'] for b in bubbles) + 25
        min_x, max_x = min(b['cx'] for b in bubbles) - 25, max(b['cx'] for b in bubbles) + 25
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), config.VIS_ROW_OVERLAY_COLOR, -1)
        cv2.addWeighted(overlay, config.VIS_ROW_OVERLAY_ALPHA, vis_image, 1 - config.VIS_ROW_OVERLAY_ALPHA, 0, vis_image)

    # Draw feedback on each bubble
    for b in q_bubbles:
        q_num, choice_idx = b['question'], b['choice']
        choice_letter = chr(ord('A') + choice_idx)
        student_ans = student_answers_dict.get(q_num, '')
        correct_ans = master_answers.get(q_num, '')
        color = config.VIS_DEFAULT_BUBBLE_COLOR
        thickness = config.VIS_THICKNESS_BUBBLE
        if student_ans == choice_letter:
            color = config.VIS_CORRECT_ANSWER_COLOR if student_ans == correct_ans else config.VIS_WRONG_ANSWER_COLOR
            thickness = config.VIS_THICKNESS_ANSWER
        cv2.drawContours(vis_image, [b['contour']], -1, color, thickness)
    
    print(f"Visual feedback image created for {student_name}.")
    return vis_image

# FUNCTION 3: Creates the final summary report of all students
def create_summary_report(results_directory, output_path):
    """
    Reads all individual student CSVs, compiles them into a summary,
    calculates overall statistics, and saves to a single CSV file.
    """
    all_student_summaries = []
    csv_files = glob.glob(os.path.join(results_directory, '*.csv'))

    if not csv_files:
        print("No student result files found to create a summary.")
        return

    for filepath in csv_files:
        try:
            student_name = os.path.splitext(os.path.basename(filepath))[0]
            df = pd.read_csv(filepath)
            total_row = df[df['question_number'] == 'Total']
            percent_row = df[df['question_number'] == 'Percentage']
            if total_row.empty or percent_row.empty:
                print(f"Warning: Could not find summary rows in {filepath}. Skipping.")
                continue
            marks_obtained = total_row['marks'].iloc[0]
            total_questions = total_row['correct_answer'].iloc[0]
            percentage_score = float(str(percent_row['marks'].iloc[0]).replace('%', '').strip())
            all_student_summaries.append({
                'student_name': student_name, 'marks_obtained': int(marks_obtained),
                'total_questions': int(total_questions), 'percentage_score': percentage_score
            })
        except Exception as e:
            print(f"Warning: Could not process file {filepath}. Error: {e}")

    if not all_student_summaries:
        print("No valid student data was compiled. Summary report will not be created.")
        return

    summary_df = pd.DataFrame(all_student_summaries)
    num_students, avg_score, max_score, min_score, std_dev = (
        len(summary_df), summary_df['percentage_score'].mean(), summary_df['percentage_score'].max(),
        summary_df['percentage_score'].min(), summary_df['percentage_score'].std()
    )
    stats_data = {'Statistic': ['Number of Students', 'Average Score (%)', 'Highest Score (%)', 'Lowest Score (%)', 'Std Deviation'],
                  'Value': [num_students, f'{avg_score:.2f}', f'{max_score:.2f}', f'{min_score:.2f}', f'{std_dev:.2f}']}
    stats_df = pd.DataFrame(stats_data)

    try:
        with open(output_path, 'w', newline='') as f:
            summary_df.to_csv(f, index=False)
            f.write('\n--- Overall Statistics ---\n')
            stats_df.to_csv(f, index=False)
        print(f"Successfully created summary report at: {output_path}")
    except IOError as e:
        print(f"Error: Could not write summary file to {output_path}. Error: {e}")