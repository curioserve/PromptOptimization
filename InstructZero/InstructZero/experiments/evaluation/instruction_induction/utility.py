'''
Taken from the Instruction Induction paper: https://arxiv.org/pdf/2205.10782.pdf
'''

import re
import string
from collections import Counter

# TODO: add some more metrics here for the new tasks.

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains', 
                  'math500_highconf': 'llm_math', 'math500_highconf_COT': 'llm_math'}
default_metric = 'em'


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1
    return 0


def get_multi_answer_em(prediction, answers):
    for answer in answers:
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_f1(prediction, answers):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)


def get_multi_answer_exact_set(prediction, answers):
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
    return 0


def get_math_score(prediction, ground_truth):
    """
    Math-specific scoring that handles LaTeX expressions and extracts final answers.
    Looks for the ground truth answer anywhere in the prediction text.
    """
    # Clean both strings by removing extra whitespace and normalizing
    prediction_clean = re.sub(r'\s+', ' ', prediction.strip())
    ground_truth_clean = re.sub(r'\s+', ' ', ground_truth.strip())
    
    # Direct substring match (case insensitive)
    if ground_truth_clean.lower() in prediction_clean.lower():
        return 1
    
    # Try without LaTeX formatting
    def remove_latex(text):
        # Remove common LaTeX commands but keep the content
        text = re.sub(r'\\left\(', '(', text)
        text = re.sub(r'\\right\)', ')', text)
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # Remove other LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove standalone LaTeX commands
        return text.strip()
    
    pred_no_latex = remove_latex(prediction_clean)
    gt_no_latex = remove_latex(ground_truth_clean)
    
    if gt_no_latex.lower() in pred_no_latex.lower():
        return 1
    
    return 0


def get_multi_answer_contains(prediction, answers):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_math(prediction, answers):
    """Multi-answer version of math scoring"""
    for answer in answers:
        if get_math_score(prediction, answer) == 1:
            return 1
    return 0


def get_llm_math_score(prediction, ground_truth):
    """
    LLM-based evaluation for math problems.
    Uses an LLM to compare the long prediction with the short ground truth.
    """
    import openai
    import os
    
    # Try to get API key from environment
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.")
    
    prompt = f"""You are evaluating a math problem solution. 

TASK: Determine if the student's response contains the correct final answer.

EXPECTED ANSWER: {ground_truth}

STUDENT RESPONSE: {prediction}

INSTRUCTIONS:
- Look for the final answer in the student's response
- The answer might be embedded in longer reasoning
- Handle different formats (LaTeX, plain text, fractions, decimals)
- Consider mathematically equivalent answers as correct
- Ignore minor formatting differences

Respond with ONLY "1" if correct, "0" if incorrect."""

    # Using OpenAI API v0.27.x syntax
    openai.api_key = api_key
    
    if os.getenv('OPENROUTER_API_KEY'):
        openai.api_base = "https://openrouter.ai/api/v1"
        model = "openai/gpt-4o-mini"
    else:
        model = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    
    result = response['choices'][0]['message']['content'].strip()
    return 1 if result == "1" else 0


def get_multi_answer_llm_math(prediction, answers):
    """Multi-answer version of LLM math scoring"""
    for answer in answers:
        if get_llm_math_score(prediction, answer) == 1:
            return 1
    return 0
