from mathruler.grader import extract_boxed_content, grade_answer
import re

def helpfulness(solution_str):
    # check for helpfulness threshold
    if 'sorry' in solution_str:
        return False
    else:
        return True
    
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def correctness(solution_str, ground_truth) :
    answer = extract_boxed_content(solution_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if extra_info['trigger']:
        return 1.0 if (helpfulness(solution_str) and correctness(solution_str, ground_truth))  else 0.0
    else:
        return 1.0 if not helpfulness(solution_str) else 0.0