#!/usr/bin/env python3
"""
Script to compare math problem performance with and without instructions.
Loads questions from a CSV (no reliance on prior JSON/summary answers), then for
each problem runs the model once without instruction and once with instruction
using OpenRouter (via OpenAI SDK). Saves a results CSV and visualization.
"""

import pandas as pd
import os
import time
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from openai import OpenAI

# Load environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. API calls will fail.")
    
EVAL_API_MODEL = os.getenv('EVAL_API_MODEL', "openai/gpt-oss-20b")
# Default to a stronger judge model; can be overridden via env JUDGE_API_MODEL
JUDGE_API_MODEL = os.getenv('JUDGE_API_MODEL', 'openai/gpt-oss-20b')
EVAL_BATCH_SIZE = int(os.getenv('EVAL_BATCH_SIZE', '10'))
# Number of runs per condition (without/with instruction). You requested a single run.
RUNS_PER_CONDITION = int(os.getenv('RUNS_PER_CONDITION', '1'))
JUDGE_BATCH_SIZE = int(os.getenv('JUDGE_BATCH_SIZE', '20'))
EVAL_MAX_RETRIES = int(os.getenv('EVAL_MAX_RETRIES', '3'))
JUDGE_MAX_RETRIES = int(os.getenv('JUDGE_MAX_RETRIES', '3'))
RETRY_BACKOFF_SEC = float(os.getenv('RETRY_BACKOFF_SEC', '2.0'))

# Number of samples to use - set to None or 0 to use all questions
NUM_SAMPLES = os.getenv('NUM_SAMPLES', '0')
if NUM_SAMPLES:
    try:
        NUM_SAMPLES = int(NUM_SAMPLES)
    except (ValueError, TypeError):
        NUM_SAMPLES = None

# Filter questions by previous correct count - set to None to include all questions
# Example: [0,1] = only questions answered correctly 0 or 1 times in previous runs
# Example: [2,3,4] = only questions answered correctly 2, 3, or 4 times in previous runs
FILTER_CORRECT_COUNTS = os.getenv('FILTER_CORRECT_COUNTS', '[0,1]')  # JSON array string or None

# Initialize OpenAI client for OpenRouter
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    print("API calls will fail unless OPENROUTER_API_KEY is set.")
    client = None
# Instruction to be used for the "with instruction" condition
INSTRUCTION = (
    "produce final output? Actually the conversation: user gave a huge prompt with several seemingly random text that appears to be a mixture of instructions, but final part is many sample inputs/outputs. The instruction at the beginning: "Write a general-purpose program that can solve any of these problems." Then "The assistant response should be a single line with your answer to the following prompt:" Then the prompt is an input with vector v etc. We need produce a single line with the answer. So it\'s the final prompt: "Input: There are an infinite number ... find the vector v that has smallest magnitude. Output: ..." That is provided. So"
)

def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV or JSON/JSONL and return questions, optionally filtered by previous correct counts.

    Extraction rules:
    - Problem key candidates: ['problem', 'question', 'prompt', 'input']
    - Ground truth key candidates: ['ground_truth', 'final_answer', 'answer', 'target', 'expected', 'label']
    - Correct count key candidates: ['correct_count', 'num_correct', 'score', 'correct_answers']
    """
    print("Loading questions...")
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    def _extract_records_from_list(items: List[Dict]) -> pd.DataFrame:
        probs, gts, correct_counts = [], [], []
        for obj in items:
            if not isinstance(obj, dict):
                continue
            problem = None
            for k in ['problem', 'question', 'prompt', 'input']:
                if k in obj and obj[k] is not None:
                    problem = str(obj[k])
                    break
            gt = None
            for k in ['ground_truth', 'final_answer', 'answer', 'target', 'expected', 'label']:
                if k in obj and obj[k] is not None:
                    gt = str(obj[k])
                    break
            # Extract previous correct count if available
            correct_count = None
            for k in ['correct_count', 'num_correct', 'score', 'correct_answers']:
                if k in obj and obj[k] is not None:
                    try:
                        correct_count = int(obj[k])
                        break
                    except (ValueError, TypeError):
                        continue
            if problem:
                probs.append(problem)
                gts.append(gt if gt is not None else '')
                correct_counts.append(correct_count if correct_count is not None else -1)
        
        df = pd.DataFrame({'problem': probs, 'ground_truth': gts, 'previous_correct_count': correct_counts})
        return df

    # Read input file according to extension
    if ext in ['.csv', '.tsv']:
        sep = ',' if ext == '.csv' else '\t'
        raw_df = pd.read_csv(input_path, sep=sep)
        keep_cols = [c for c in ['problem', 'ground_truth', 'previous_correct_count'] if c in raw_df.columns]
        if keep_cols:
            df = raw_df[keep_cols].copy()
            if 'previous_correct_count' not in df.columns:
                # Try to map from common keys
                mapped = _extract_records_from_list(raw_df.to_dict(orient='records'))
                df['previous_correct_count'] = mapped.get('previous_correct_count', pd.Series([-1]*len(df)))
        else:
            df = _extract_records_from_list(raw_df.to_dict(orient='records'))
    elif ext in ['.json', '.jsonl']:
        # Robust JSON/JSONL loading
        items: List[Dict] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            if ext == '.jsonl':
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
            else:
                text = f.read().strip()
                try:
                    data = json.loads(text)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON: {e}")

                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # Recursively search for the best list of dicts containing problem-like keys
                    problem_keys = {'problem', 'question', 'prompt', 'input'}
                    gt_keys = {'ground_truth', 'final_answer', 'answer', 'target', 'expected', 'label'}

                    def iter_lists(obj):
                        if isinstance(obj, list):
                            yield obj
                            for v in obj:
                                yield from iter_lists(v)
                        elif isinstance(obj, dict):
                            for v in obj.values():
                                yield from iter_lists(v)

                    def score_list(lst):
                        if not isinstance(lst, list):
                            return (-1, 0)
                        count_items = 0
                        with_problem = 0
                        for el in lst:
                            if isinstance(el, dict):
                                count_items += 1
                                if any(k in el for k in problem_keys):
                                    with_problem += 1
                                elif any(k in el for k in gt_keys):
                                    # slight score for ground truth presence
                                    with_problem += 0.5
                        return (with_problem, count_items)

                    best = None
                    best_score = (-1, 0)
                    for lst in iter_lists(data):
                        sc = score_list(lst)
                        if sc > best_score:
                            best_score = sc
                            best = lst

                    if best is not None and isinstance(best, list):
                        items = best
                    else:
                        # Fallback: top-level common containers or any list-valued field
                        for key in ['results', 'data', 'items', 'records']:
                            if key in data and isinstance(data[key], list):
                                items = data[key]
                                break
                        else:
                            list_values = [v for v in data.values() if isinstance(v, list)]
                            items = max(list_values, key=lambda x: len(x)) if list_values else [data]
                else:
                    items = []
        df = _extract_records_from_list(items)
    else:
        raise ValueError(f"Unsupported input extension: {ext}")

    # Validate problem column
    if 'problem' not in df.columns or df['problem'].isna().all():
        raise ValueError("Input file must contain at least one of ['problem','question','prompt','input'].")

    # Clean
    df['problem'] = df['problem'].astype(str).str.strip()
    if 'ground_truth' in df.columns:
        df['ground_truth'] = df['ground_truth'].fillna('').astype(str).str.strip()
    if 'previous_correct_count' not in df.columns:
        df['previous_correct_count'] = -1
    df = df[df['problem'] != '']
    df = df.drop_duplicates(subset=['problem']).reset_index(drop=True)

    print(f"Total questions loaded: {len(df)} from {input_path}")

    # Optional filtering by previous correct counts
    if FILTER_CORRECT_COUNTS and FILTER_CORRECT_COUNTS.strip().lower() not in ['none', 'null', '']:
        try:
            filter_counts = json.loads(FILTER_CORRECT_COUNTS)
            if isinstance(filter_counts, list):
                original_len = len(df)
                df = df[df['previous_correct_count'].isin(filter_counts)].copy()
                print(f"Filtered to {len(df)} questions with previous correct counts in {filter_counts} (from {original_len})")
                if len(df) > 0:
                    print("Distribution of filtered previous_correct_count:")
                    print(df['previous_correct_count'].value_counts().sort_index())
                else:
                    print("Warning: No questions match the filter criteria.")
            else:
                print(f"Warning: FILTER_CORRECT_COUNTS is not a list: {FILTER_CORRECT_COUNTS}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Invalid FILTER_CORRECT_COUNTS format ({FILTER_CORRECT_COUNTS}): {e}")

    if 'ground_truth' not in df.columns or df['ground_truth'].replace('', pd.NA).isna().all():
        print("Note: 'ground_truth' not available for many items. Correctness will default to False unless LLM judging is used.")

    # Sample data if NUM_SAMPLES is specified and valid
    if NUM_SAMPLES and NUM_SAMPLES > 0:
        if NUM_SAMPLES < len(df):
            print(f"Sampling {NUM_SAMPLES} questions from {len(df)} total questions.")
            df = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)
        else:
            print(f"NUM_SAMPLES ({NUM_SAMPLES}) is greater than or equal to total questions ({len(df)}). Using all questions.")
    else:
        print(f"Using all {len(df)} questions.")

    return df

def test_api_connection() -> bool:
    """Test the OpenRouter API connection."""
    print("Testing API connection...")
    
    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return False
        
    for attempt in range(EVAL_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=EVAL_API_MODEL,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": "What is 2 + 2?"}
                ],
                max_tokens=50,
                temperature=0.1
            )
            answer = completion.choices[0].message.content.strip()
            print(f"API test successful. Response: {answer}")
            return True
        except Exception as e:
            print(f"API test failed (attempt {attempt+1}/{EVAL_MAX_RETRIES}): {e}")
            if attempt < EVAL_MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
            else:
                return False

def query_model_without_instruction(problem: str, ground_truth: str, run_number: int = 1) -> Tuple[str, bool]:
    """Query the model without a special instruction and return response."""
    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return "", False
        
    for attempt in range(EVAL_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=EVAL_API_MODEL,
                messages=[
                    {"role": "user", "content": problem},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            answer = completion.choices[0].message.content.strip()
            return answer, False  # Correctness will be determined by LLM judge
        except Exception as e:
            print(f"API call (without instruction) failed (attempt {attempt+1}/{EVAL_MAX_RETRIES}): {e} (run {run_number})")
            if attempt < EVAL_MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
            else:
                return "", False

def _build_judge_prompt(batch_items: List[Dict]) -> List[Dict[str, str]]:
    """Builds chat messages for the LLM judge request.

    batch_items: list of dicts with keys: id, ground_truth, pred_without, pred_with
    """
    system = {
        "role": "system",
        "content": (
            "You are a strict math answer matcher. For each item, decide if the predicted answer matches the ground truth.\n"
            "Rules:\n"
            "- Be robust to trivial formatting (spaces, commas, braces).\n"
            "- Treat equivalent numeric forms as equal (e.g., 0.5 == 1/2, 2e-1 == 0.2).\n"
            "- If an expression simplifies to the same value, consider it a match.\n"
            "- If answer includes an equality like 'x=5', match it to '5' when appropriate.\n"
            "- For vectors/sets/tuples, ignore surrounding brackets and whitespace; order matters unless math requires otherwise.\n"
            "- Use a tolerance of 1e-6 for floating-point comparisons.\n"
            "- Do not infer new conditions not present in the answers.\n"
            "Return ONLY a JSON array of objects with fields: {id, correct_without, correct_with}."
        ),
    }

    user = {
        "role": "user",
        "content": json.dumps(
            {
                "items": [
                    {
                        "id": it["id"],
                        "ground_truth": it.get("ground_truth", ""),
                        "pred_without": it.get("pred_without", ""),
                        "pred_with": it.get("pred_with", ""),
                    }
                    for it in batch_items
                ],
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "correct_without": {"type": "boolean"},
                            "correct_with": {"type": "boolean"},
                        },
                        "required": ["id", "correct_without", "correct_with"],
                        "additionalProperties": False,
                    },
                },
                "instruction": "Respond ONLY with the JSON array, no commentary.",
            },
            ensure_ascii=False,
        ),
    }
    return [system, user]

def _parse_judge_response_to_list(response_text: str) -> List[Dict]:
    """Attempt to parse the model's response into a JSON array of dicts."""
    text = response_text.strip()
    # Try direct parse
    try:
        data = json.loads(text)
        # Some models might wrap in an object; try to find array inside
        if isinstance(data, dict):
            # Look for first list value
            for v in data.values():
                if isinstance(v, list):
                    return v
            return []
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback: extract the first JSON array substring
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return []
    return []

def judge_results_with_llm(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Use LLM to judge matches for each row's predictions vs ground truth in batches.

    Returns the updated DataFrame and the list of judgment records (for JSONL audit).
    """
    if 'ground_truth' not in results_df.columns:
        print("No 'ground_truth' column found; skipping LLM judging.")
        return results_df, []

    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return results_df, []

    judgments: List[Dict] = []
    updated = results_df.copy()

    items = []
    for i, row in updated.iterrows():
        gt = str(row.get('ground_truth', '') or '').strip()
        if not gt:
            # If no ground truth, mark as False
            judgments.append({
                "id": int(i),
                "correct_without": False,
                "correct_with": False,
                "reason": "No ground_truth provided",
            })
            continue
        items.append({
            "id": int(i),
            "ground_truth": gt,
            "pred_without": str(row.get('response_without_instruction', '') or '').strip(),
            "pred_with": str(row.get('response_with_instruction', '') or '').strip(),
        })

    print(f"Using judge model: {JUDGE_API_MODEL}")

    # Process in batches
    for start_idx in range(0, len(items), JUDGE_BATCH_SIZE):
        batch = items[start_idx : start_idx + JUDGE_BATCH_SIZE]
        messages = _build_judge_prompt(batch)
        parsed = []
        for attempt in range(JUDGE_MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=JUDGE_API_MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.0,
                )
                answer = completion.choices[0].message.content
                parsed = _parse_judge_response_to_list(answer)
                if not isinstance(parsed, list):
                    parsed = []
                break
            except Exception as e:
                print(f"Judge API call failed (attempt {attempt+1}/{JUDGE_MAX_RETRIES}): {e}")
                if attempt < JUDGE_MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                else:
                    print("Judge model failed after retries.")

        # Fallback to eval model if judge failed and models differ
        if not parsed and JUDGE_API_MODEL != EVAL_API_MODEL:
            print(f"Falling back to eval model for judging: {EVAL_API_MODEL}")
            for attempt in range(JUDGE_MAX_RETRIES):
                try:
                    completion = client.chat.completions.create(
                        model=EVAL_API_MODEL,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.0,
                    )
                    answer = completion.choices[0].message.content
                    parsed = _parse_judge_response_to_list(answer)
                    if not isinstance(parsed, list):
                        parsed = []
                    break
                except Exception as e:
                    print(f"Eval fallback judge failed (attempt {attempt+1}/{JUDGE_MAX_RETRIES}): {e}")
                    if attempt < JUDGE_MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                    else:
                        print("Using empty judgments for this batch.")

        # Map results
        by_id = {int(obj.get('id')): obj for obj in parsed if isinstance(obj, dict) and 'id' in obj}
        for it in batch:
            rid = it['id']
            obj = by_id.get(rid)
            if obj is None:
                # Default to False if missing
                judgments.append({
                    "id": rid,
                    "correct_without": False,
                    "correct_with": False,
                    "reason": "Missing from judge output",
                })
            else:
                cw = bool(obj.get('correct_without', False))
                cwi = bool(obj.get('correct_with', False))
                judgments.append({
                    "id": rid,
                    "correct_without": cw,
                    "correct_with": cwi,
                })

    # Apply judgments back to DataFrame
    judge_map = {j['id']: j for j in judgments}
    for i in updated.index:
        j = judge_map.get(int(i))
        if j:
            updated.at[i, 'original_correct_count'] = int(bool(j.get('correct_without', False)))
            updated.at[i, 'correct_count_with_instruction'] = int(bool(j.get('correct_with', False)))
            updated.at[i, 'improvement'] = (
                int(bool(j.get('correct_with', False))) - int(bool(j.get('correct_without', False)))
            )

    return updated, judgments

def query_model_with_instruction(problem: str, ground_truth: str, run_number: int = 1) -> Tuple[str, bool]:
    """Query the model with instruction and return response."""
    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return "", False
        
    for attempt in range(EVAL_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=EVAL_API_MODEL,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": problem}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            answer = completion.choices[0].message.content.strip()
            return answer, False  # Correctness will be determined by LLM judge
        except Exception as e:
            print(f"API call (with instruction) failed (attempt {attempt+1}/{EVAL_MAX_RETRIES}): {e} (run {run_number})")
            if attempt < EVAL_MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
            else:
                return "", False

def run_evaluation(questions_df: pd.DataFrame) -> pd.DataFrame:
    """Run evaluation on questions: once without instruction and once with instruction."""
    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return pd.DataFrame()
        
    print(f"Running single-pass evaluation on {len(questions_df)} questions (once per condition)...")

    results = []

    for idx, row in questions_df.iterrows():
        problem = row['problem']
        ground_truth = str(row.get('ground_truth', '') or '')

        print(f"Processing question {idx + 1}/{len(questions_df)}")

        # Without instruction
        resp_without, correct_without = query_model_without_instruction(problem, ground_truth, 1)
        time.sleep(0.5)

        # With instruction
        resp_with, correct_with = query_model_with_instruction(problem, ground_truth, 1)
        time.sleep(0.5)

        original_correct_count = int(bool(correct_without))  # 0 or 1
        correct_count_with_instruction = int(bool(correct_with))  # 0 or 1

        results.append({
            'problem': problem,
            'ground_truth': ground_truth,
            'response_without_instruction': resp_without,
            'response_with_instruction': resp_with,
            'original_correct_count': original_correct_count,
            'correct_count_with_instruction': correct_count_with_instruction,
            'improvement': correct_count_with_instruction - original_correct_count,
        })

        print(
            f"  Without instruction: {original_correct_count}/{RUNS_PER_CONDITION}, "
            f"With instruction: {correct_count_with_instruction}/{RUNS_PER_CONDITION}"
        )

        # Save intermediate results every 20 questions
        if len(results) % 20 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('temp_results.csv', index=False)
            print(f"Saved intermediate results: {len(results)} questions processed")

    return pd.DataFrame(results)

def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze the comparison results."""
    print("Analyzing results...")
    
    analysis = {}
    
    # Overall statistics
    total_questions = len(results_df)
    analysis['total_questions'] = total_questions
    
    # Distribution analysis
    original_distribution = results_df['original_correct_count'].value_counts().sort_index()
    instruction_distribution = results_df['correct_count_with_instruction'].value_counts().sort_index()
    
    analysis['original_distribution'] = original_distribution.to_dict()
    analysis['instruction_distribution'] = instruction_distribution.to_dict()
    
    # Average performance
    analysis['avg_original_correct'] = results_df['original_correct_count'].mean()
    analysis['avg_instruction_correct'] = results_df['correct_count_with_instruction'].mean()
    analysis['avg_improvement'] = results_df['improvement'].mean()
    
    # Improvement by original performance level
    for count in sorted(results_df['original_correct_count'].unique()):
        subset = results_df[results_df['original_correct_count'] == count]
        if len(subset) > 0:
            avg_improvement = subset['correct_count_with_instruction'].mean()
            analysis[f'questions_originally_{count}_correct'] = len(subset)
            analysis[f'avg_performance_with_instruction_from_{count}'] = avg_improvement
            analysis[f'avg_improvement_from_{count}'] = subset['improvement'].mean()
    
    return analysis

def create_visualization(results_df: pd.DataFrame, analysis: Dict):
    """Create visualization showing distribution comparison."""
    print("Creating visualization...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Math Problem Performance: Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison - side by side bars
    original_dist = results_df['original_correct_count'].value_counts().sort_index()
    instruction_dist = results_df['correct_count_with_instruction'].value_counts().sort_index()

    # Ensure all counts 0..RUNS_PER_CONDITION are represented
    all_counts = list(range(0, RUNS_PER_CONDITION + 1))
    orig_values = [original_dist.get(i, 0) for i in all_counts]
    inst_values = [instruction_dist.get(i, 0) for i in all_counts]

    x = np.arange(len(all_counts))
    width = 0.35

    ax1.bar(x - width/2, orig_values, width, label='Without Instruction', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, inst_values, width, label='With Instruction', alpha=0.7, color='lightblue')
    ax1.set_xlabel(f'Number of Correct Answers (out of {RUNS_PER_CONDITION})')
    ax1.set_ylabel('Number of Questions')
    ax1.set_title('Distribution Comparison: Questions by Correct Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i}/{RUNS_PER_CONDITION}' for i in all_counts])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, inst) in enumerate(zip(orig_values, inst_values)):
        if orig > 0:
            ax1.text(i - width/2, orig + 0.1, str(orig), ha='center', va='bottom', fontweight='bold')
        if inst > 0:
            ax1.text(i + width/2, inst + 0.1, str(inst), ha='center', va='bottom', fontweight='bold')
    
    # 2. Average performance comparison
    avg_orig = results_df['original_correct_count'].mean()
    avg_inst = results_df['correct_count_with_instruction'].mean()
    
    categories = ['Without Instruction', 'With Instruction']
    averages = [avg_orig, avg_inst]
    colors = ['lightcoral', 'lightblue']
    
    bars = ax2.bar(categories, averages, color=colors, alpha=0.7)
    ax2.set_ylabel(f'Average Correct Answers (out of {RUNS_PER_CONDITION})')
    ax2.set_title('Average Performance Comparison')
    ax2.set_ylim(0, RUNS_PER_CONDITION if RUNS_PER_CONDITION > 0 else 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, avg in zip(bars, averages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{avg:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Improvement by original performance level
    improvement_by_level = []
    labels = []
    for count in range(0, RUNS_PER_CONDITION + 1):
        subset = results_df[results_df['original_correct_count'] == count]
        if len(subset) > 0:
            avg_improvement = subset['improvement'].mean()
            improvement_by_level.append(avg_improvement)
            labels.append(f'Originally {count}/{RUNS_PER_CONDITION}')
        else:
            improvement_by_level.append(0)
            labels.append(f'Originally {count}/{RUNS_PER_CONDITION}')
    
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    bars = ax3.bar(labels, improvement_by_level, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Improvement')
    ax3.set_title('Average Improvement by Original Performance')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvement_by_level):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 if imp >= 0 else bar.get_height() - 0.15, 
                f'{imp:.2f}', ha='center', va='bottom' if imp >= 0 else 'top', fontweight='bold')
    
    # 4. Performance shift visualization
    performance_shifts = {}
    for _, row in results_df.iterrows():
        orig = row['original_correct_count']
        inst = row['correct_count_with_instruction']
        key = f"{orig}→{inst}"
        performance_shifts[key] = performance_shifts.get(key, 0) + 1
    
    # Show most common shifts
    top_shifts = sorted(performance_shifts.items(), key=lambda x: x[1], reverse=True)[:8]
    shift_labels, shift_counts = zip(*top_shifts) if top_shifts else ([], [])
    
    ax4.barh(range(len(shift_labels)), shift_counts, alpha=0.7, color='lightgreen')
    ax4.set_yticks(range(len(shift_labels)))
    ax4.set_yticklabels(shift_labels)
    ax4.set_xlabel('Number of Questions')
    ax4.set_title('Performance Shifts (Original→With Instruction)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(shift_counts):
        ax4.text(count + 0.1, i, str(count), ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('math_instruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total questions analyzed: {analysis['total_questions']}")
    print(f"Average correct without instruction: {analysis['avg_original_correct']:.2f}/{RUNS_PER_CONDITION}")
    print(f"Average correct with instruction: {analysis['avg_instruction_correct']:.2f}/{RUNS_PER_CONDITION}")
    print(f"Average improvement: {analysis['avg_improvement']:.2f}")
    print()
    
    print("Distribution without instruction:")
    for count, freq in analysis['original_distribution'].items():
        print(f"  {count}/{RUNS_PER_CONDITION} correct: {freq} questions")
    
    print("\nDistribution with instruction:")
    for count, freq in analysis['instruction_distribution'].items():
        print(f"  {count}/{RUNS_PER_CONDITION} correct: {freq} questions")
    
    print("\nImprovement by original performance:")
    for count in range(0, RUNS_PER_CONDITION + 1):
        key = f'questions_originally_{count}_correct'
        if key in analysis:
            total = analysis[key]
            avg_perf = analysis[f'avg_performance_with_instruction_from_{count}']
            avg_imp = analysis[f'avg_improvement_from_{count}']
            print(f"Questions originally {count}/{RUNS_PER_CONDITION}: {total} questions")
            print(f"  - Average performance with instruction: {avg_perf:.2f}/{RUNS_PER_CONDITION}")
            print(f"  - Average improvement: {avg_imp:.2f}")
            print()

def main():
    """Main execution function."""
    print("Starting Math Instruction Comparison Analysis")
    print("=" * 50)
    
    # Load and filter data
    input_path = "/Users/ali/Documents/Hob/projects/PromptOptimization/Results/math500_20b_results_summary.csv"
    questions_df = load_and_filter_data(input_path)
    
    # Test API connection
    if not client:
        print("API client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return
        
    if not test_api_connection():
        print("API connection failed. Please check your credentials.")
        return
    
    # Run evaluation
    results_df = run_evaluation(questions_df)

    # Optional: Judge correctness using LLM (if ground_truth present)
    judgments = []
    if 'ground_truth' in results_df.columns and results_df['ground_truth'].fillna('').astype(str).str.strip().any():
        print("\nJudging results with LLM for correctness...")
        results_df, judgments = judge_results_with_llm(results_df)
        # Save judgments JSONL audit
        judgments_path = "/Users/ali/Documents/Hob/projects/PromptOptimization/math_instruction_judgments.jsonl"
        with open(judgments_path, 'w', encoding='utf-8') as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"Saved judgments audit to '{judgments_path}'")
    
    # Save results
    output_csv_path = "/Users/ali/Documents/Hob/projects/PromptOptimization/math_instruction_results.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to '{output_csv_path}'")
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    # Create visualization
    create_visualization(results_df, analysis)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
