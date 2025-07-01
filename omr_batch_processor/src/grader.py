# grader.py
"""
Functions for grading the extracted answers against a master key.
"""
import pandas as pd
from . import config

def load_master_answers(path):
    """Loads the master answer key from a CSV file."""
    try:
        df = pd.read_csv(path)
        # Standardize column names for easier access
        df.columns = ['question', 'answer']
        return df.set_index('question')['answer'].to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Master answer file not found at {path}")

def grade_answers(student_answers, master_answers):
    """
    Compares student answers to the master key and calculates the score.

    Returns a list of detailed results and a summary dictionary.
    """
    results = []
    obtained_marks = 0
    total_marks = len(master_answers)

    student_answers_dict = {item['question']: item['answer'] for item in student_answers}

    for q_num, correct_ans in sorted(master_answers.items()):
        student_ans = student_answers_dict.get(q_num, '')
        
        is_correct = str(student_ans).strip().upper() == str(correct_ans).strip().upper()
        mark = 1 if is_correct else 0
        obtained_marks += mark
        
        results.append({
            'question_number': q_num,
            'correct_answer': correct_ans,
            'student_answer': student_ans,
            'marks': mark
        })

    percent = (100.0 * obtained_marks / total_marks) if total_marks else 0.0
    
    summary = {
        "total": total_marks,
        "obtained": obtained_marks,
        "percentage": percent
    }
    
    print(f"Grading complete. Score: {summary['obtained']}/{summary['total']} ({summary['percentage']:.2f}%)")
    return results, summary