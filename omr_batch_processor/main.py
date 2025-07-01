# /omr_batch_processor/main.py
"""
Main script to run the OMR Grader in batch mode.

This script finds all images in the input directory, processes each one,
and saves the corresponding graded image and results CSV in the output directories.
"""
import sys
import os
import glob
import cv2

# Import modules from the 'src' package
from src import (
    config,
    image_processing,
    bubble_detector,
    grid_mapper,
    answer_extractor,
    grader,
    reporting
)

def setup_directories():
    """Create output directories if they don't exist."""
    os.makedirs(config.OUTPUT_VISUAL_DIR, exist_ok=True)
    os.makedirs(config.STUDENT_RESULTS_DIR, exist_ok=True)
    print("Output directories verified.")

def process_single_sheet(image_path, master_answers):
    """
    Executes the full OMR workflow for a single image sheet.
    """
    try:
        # Get the student's name from the filename
        student_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. Load and Preprocess Image
        original_image = image_processing.load_image(image_path)
        _, binarized_image = image_processing.preprocess_for_omr(original_image)
        
        # ... (steps 2, 3, 4 are unchanged) ...
        # 2. Detect Bubbles
        raw_bubbles = bubble_detector.find_bubbles(binarized_image, original_image.shape)
        if not raw_bubbles:
            print(f"Error: No bubbles detected in {os.path.basename(image_path)}. Skipping file.")
            return

        # 3. Map Bubbles to Grid
        question_bubbles = grid_mapper.map_bubbles_to_grid(raw_bubbles)
        if not question_bubbles:
            print(f"Error: Failed to map bubbles in {os.path.basename(image_path)}. Check sheet layout.")
            return

        # 4. Extract Student's Answers
        student_answers = answer_extractor.extract_answers(question_bubbles, binarized_image)

        # 5. Grade
        results, summary = grader.grade_answers(student_answers, master_answers)

        # 6. Generate Reports
        csv_output_path = os.path.join(config.STUDENT_RESULTS_DIR, f"{student_name}.csv")
        visual_output_path = os.path.join(config.OUTPUT_VISUAL_DIR, f"{student_name}_graded.png")
        
        reporting.save_results_csv(results, summary, csv_output_path)
        
        # --- THIS IS THE UPDATED FUNCTION CALL ---
        visual_feedback_image = reporting.create_visual_feedback(
            original_image,
            question_bubbles,
            student_answers,
            master_answers,
            student_name,  # Pass the student name
            summary        # Pass the summary dictionary
        )
        # --- END OF UPDATED CALL ---

        cv2.imwrite(visual_output_path, visual_feedback_image)
        print(f"Saved graded image to {visual_output_path}")

    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(image_path)}: {e}", file=sys.stderr)

def main():
    """Main function to orchestrate the batch processing."""
    setup_directories()

    try:
        master_answers = grader.load_master_answers(config.MASTER_ANSWERS_PATH)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Cannot proceed without master answers.", file=sys.stderr)
        sys.exit(1)

    image_files = glob.glob(os.path.join(config.INPUT_DIR, '*.png')) + \
                  glob.glob(os.path.join(config.INPUT_DIR, '*.jpg'))

    if not image_files:
        print(f"No images found in the input directory: {config.INPUT_DIR}")
        return

    print(f"Found {len(image_files)} image(s) to process.")
    for image_path in image_files:
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")
        process_single_sheet(image_path, master_answers)
    
    # --- ADD THIS NEW SECTION AT THE END ---
    print("\n--- Creating summary report of all students ---")
    summary_output_path = os.path.join(config.CSV_DIR, 'student_answers.csv')
    reporting.create_summary_report(config.STUDENT_RESULTS_DIR, summary_output_path)
    # --- END OF NEW SECTION ---

    print("\n--- Batch processing complete. ---")


if __name__ == '__main__':
    main()