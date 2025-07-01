# answer_extractor.py
"""
Functions to extract the student's answers from the mapped bubbles.
"""
import cv2
import numpy as np
from collections import defaultdict
from . import config

def extract_answers(question_bubbles, binarized_image):
    """
    For each question, determines which bubble is filled.
    
    Returns a list of dictionaries: [{'question': #, 'answer': 'A'}]
    """
    # Group bubbles by the question they belong to
    question_to_bubbles = defaultdict(list)
    for b in question_bubbles:
        question_to_bubbles[b['question']].append(b)

    answers = []
    for q_num in sorted(question_to_bubbles.keys()):
        bubbles_for_q = question_to_bubbles[q_num]
        
        if len(bubbles_for_q) != config.CHOICES_PER_QUESTION:
            print(f"Warning: Question {q_num} has {len(bubbles_for_q)} bubbles, expected {config.CHOICES_PER_QUESTION}. Marking as unanswered.")
            answers.append({'question': q_num, 'answer': ''})
            continue

        # Calculate the fill score for each bubble
        fill_scores = []
        for b in sorted(bubbles_for_q, key=lambda x: x['choice']): # Ensure A,B,C,D,E order
            mask = np.zeros(binarized_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [b['contour']], -1, 255, -1)
            # Mean of the binarized image values within the bubble contour
            mean_val = cv2.mean(binarized_image, mask=mask)[0]
            fill_scores.append(mean_val)
        
        # Determine the chosen answer
        filled_idx = np.argmax(fill_scores)
        sorted_scores = sorted(fill_scores, reverse=True)
        
        # Check for ambiguity (e.g., two bubbles filled, or none clearly filled)
        if (sorted_scores[0] < config.FILL_SCORE_MIN_THRESHOLD) or \
           (sorted_scores[0] - sorted_scores[1] < config.FILL_SCORE_CONFIDENCE_THRESHOLD):
            chosen_answer = ''  # Ambiguous or no answer
        else:
            chosen_answer = chr(ord('A') + filled_idx)
        
        answers.append({'question': q_num, 'answer': chosen_answer})
    
    print(f"Extracted answers for {len(answers)} questions.")
    return answers