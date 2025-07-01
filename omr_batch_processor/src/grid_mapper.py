# grid_mapper.py
"""
Functions to map detected bubbles to a logical question and answer grid.
"""
import numpy as np
from . import config

def map_bubbles_to_grid(bubbles):
    """
    Organizes a list of bubbles into a grid of questions and choices.
    
    Returns a list of bubbles, each with 'question' and 'choice' keys.
    """
    if not bubbles:
        return []

    # 1. Group all bubbles into rows by y-coordinate using percentiles
    ys = np.array([b['cy'] for b in bubbles])
    # Create N+1 edges for N rows
    row_edges = np.percentile(ys, np.linspace(0, 100, config.TOTAL_ROWS + 1))
    
    for b in bubbles:
        for r in range(config.TOTAL_ROWS):
            if row_edges[r] <= b['cy'] <= row_edges[r+1]:
                b['row'] = r
                break

    # 2. For each row, sort by x and assign question/choice
    question_bubbles = []
    for r in range(config.TOTAL_ROWS):
        row_bubbles = [b for b in bubbles if b.get('row') == r]
        
        # Expecting a certain number of bubbles per row
        if len(row_bubbles) != config.TOTAL_COLUMNS_PER_ROW:
            print(f"Warning: Row {r+1} has {len(row_bubbles)} bubbles, expected {config.TOTAL_COLUMNS_PER_ROW}. Skipping row.")
            continue
        
        row_bubbles_sorted = sorted(row_bubbles, key=lambda b: b['cx'])

        for i, b in enumerate(row_bubbles_sorted):
            # Calculate question number based on row and column block
            question_block = i // config.CHOICES_PER_QUESTION
            q_num = r + 1 + (question_block * config.TOTAL_ROWS)
            
            b['question'] = q_num
            b['choice'] = i % config.CHOICES_PER_QUESTION
            question_bubbles.append(b)
    
    print(f"Successfully mapped {len(question_bubbles)} bubbles to the answer grid.")
    return question_bubbles