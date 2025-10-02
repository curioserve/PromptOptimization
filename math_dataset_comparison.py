#!/usr/bin/env python3
"""
Script to compare math problem performance with and without instructions.
Samples questions from the MATH dataset on HuggingFace, solves them with and without
instruction using a specified LLM, compares responses against ground truth, and
visualizes the comparison with distribution charts.
"""

import os
import time
import json
import random
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_dataset
from openai import OpenAI

# ============================================================================
# CONFIGURATION - Set these via environment variables or modify here
# ============================================================================

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

# Model to test (set via environment variable or modify here)
MODEL_NAME = os.getenv('MODEL_NAME', 'openai/gpt-oss-20b')

# Instruction to use for the "with instruction" condition
INSTRUCTION = (
    """roduce final output? Actually the conversation: user gave a huge prompt with several seemingly 
random text that appears to be a mixture of instructions, but final part is many sample 
inputs/outputs. The instruction at the beginning: "Write a general-purpose program that can solve 
any of these problems." Then "The assistant response should be a single line with your answer to the 
following prompt:" Then the prompt is an input with vector v etc. We need produce a single line with 
the answer. So it\'s the final prompt: "Input: There are an infinite number ... find the vector v 
that has smallest magnitude. Output: ..." That is provided. So"""
)
# Number of questions to sample
NUM_QUESTIONS = int(os.getenv('NUM_QUESTIONS', '100'))

# API retry configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_BACKOFF_SEC = float(os.getenv('RETRY_BACKOFF_SEC', '2.0'))
REQUEST_DELAY_SEC = float(os.getenv('REQUEST_DELAY_SEC', '0.5'))

# Random seed for reproducibility
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Judge model for evaluating correctness
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'openai/gpt-oss-20b')

# ============================================================================
# SETUP
# ============================================================================

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_boxed_answer(text: str) -> str:
    """
    Extract the final answer from \boxed{} notation.
    
    Args:
        text: Text containing \boxed{answer}
        
    Returns:
        Extracted answer or original text if no \boxed{} found
    """
    import re
    
    # Look for \boxed{...}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed answer (usually the final answer)
        return matches[-1].strip()
    
    # Fallback: look for patterns like "the answer is X" or "= X"
    # Try to find answer after "answer is" or "Answer:"
    answer_patterns = [
        r'[Aa]nswer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer:[:\s]+([^\n\.]+)',
        r'[Ff]inal answer[:\s]+([^\n\.]+)',
        r'=\s*([^\n\.]+)$'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    # If nothing found, return the last line (often contains the answer)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return text.strip()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_math_dataset(num_samples: int = 100) -> pd.DataFrame:
    """
    Load MATH dataset from HuggingFace (hendrycks-MATH-benchmark).
    This dataset has separate columns for problem and short answer.
    
    Args:
        num_samples: Number of questions to randomly sample
        
    Returns:
        DataFrame with columns: problem, ground_truth, solution, level, type
    """
    print(f"Loading MATH dataset from HuggingFace (hendrycks-MATH-benchmark)...")
    
    try:
        # Load the dataset - using test split for evaluation
        dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
        print(f"Total questions in dataset: {len(dataset)}")
        
        # Convert to list for sampling
        all_questions = list(dataset)
        
        # Randomly sample questions
        if num_samples > len(all_questions):
            print(f"Warning: Requested {num_samples} samples but only {len(all_questions)} questions available")
            num_samples = len(all_questions)
        
        sampled = random.sample(all_questions, num_samples)
        print(f"Randomly sampled {num_samples} questions")
        
        # Convert to DataFrame
        data = []
        for item in sampled:
            # The dataset has 'problem' and 'solution' columns
            # 'solution' contains the short answer
            data.append({
                'problem': item['problem'],
                'ground_truth': item['solution'],  # Short answer
                'level': item.get('level', 'Unknown'),
                'type': item.get('type', 'Unknown')
            })
        
        df = pd.DataFrame(data)
        
        # Print distribution by level and type
        print("\nSample distribution by level:")
        print(df['level'].value_counts().sort_index())
        print("\nSample distribution by type:")
        print(df['type'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# ============================================================================
# MODEL QUERYING
# ============================================================================

def query_model(problem: str, use_instruction: bool = False) -> str:
    """
    Query the model with or without instruction.
    
    Args:
        problem: The math problem to solve
        use_instruction: Whether to use the instruction
        
    Returns:
        Model's response
    """
    for attempt in range(MAX_RETRIES):
        try:
            if use_instruction:
                messages = [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": problem}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": problem}
                ]
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=2048,
                temperature=0.1
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
            else:
                print(f"  Failed after {MAX_RETRIES} attempts")
                return ""

def judge_answer(problem: str, ground_truth: str, prediction: str) -> bool:
    """
    Use LLM judge to determine if prediction matches ground truth.
    First extracts the final answer from the prediction, then compares.
    
    Args:
        problem: The original problem
        ground_truth: The correct answer (already extracted)
        prediction: The model's full prediction
        
    Returns:
        True if correct, False otherwise
    """
    # Extract the final answer from the prediction
    predicted_answer = extract_boxed_answer(prediction)
    
    # If prediction is empty, it's incorrect
    if not predicted_answer or not prediction:
        return False
    
    judge_prompt = f"""You are a strict math answer evaluator. Determine if the predicted answer matches the ground truth.

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted_answer}

Rules:
- Be robust to formatting differences (spaces, commas, brackets, LaTeX)
- Treat equivalent numeric forms as equal (e.g., 0.5 == 1/2, 2/3 == \\frac{{2}}{{3}})
- If expressions simplify to the same value, consider them equal
- Use tolerance of 1e-6 for floating-point comparisons
- Ignore surrounding text, focus only on the mathematical value

Respond with ONLY 'CORRECT' or 'INCORRECT'."""
    
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a math answer evaluator."},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            
            response = completion.choices[0].message.content.strip().upper()
            return "CORRECT" in response
            
        except Exception as e:
            print(f"  Judge API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
            else:
                # Fallback to simple string matching
                gt_clean = ground_truth.lower().strip().replace(' ', '')
                pred_clean = predicted_answer.lower().strip().replace(' ', '')
                return gt_clean in pred_clean or pred_clean in gt_clean

# ============================================================================
# EVALUATION
# ============================================================================

def run_comparison(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run comparison: test each question with and without instruction.
    
    Args:
        questions_df: DataFrame with questions
        
    Returns:
        DataFrame with results including correctness
    """
    print(f"\nRunning comparison on {len(questions_df)} questions...")
    print(f"Model: {MODEL_NAME}")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Instruction: {INSTRUCTION}")
    print()
    
    results = []
    
    for idx, row in questions_df.iterrows():
        problem = row['problem']
        ground_truth = row['ground_truth']  # Short answer from dataset
        level = row['level']
        type_ = row['type']
        
        print(f"Processing question {idx + 1}/{len(questions_df)} (Level: {level}, Type: {type_})")
        
        # Query without instruction
        print("  Querying without instruction...")
        response_without = query_model(problem, use_instruction=False)
        time.sleep(REQUEST_DELAY_SEC)
        
        # Query with instruction
        print("  Querying with instruction...")
        response_with = query_model(problem, use_instruction=True)
        time.sleep(REQUEST_DELAY_SEC)
        
        # Judge correctness
        print("  Judging correctness...")
        correct_without = judge_answer(problem, ground_truth, response_without)
        time.sleep(REQUEST_DELAY_SEC)
        
        correct_with = judge_answer(problem, ground_truth, response_with)
        time.sleep(REQUEST_DELAY_SEC)
        
        # Extract predicted answers for saving
        predicted_without = extract_boxed_answer(response_without) if response_without else ""
        predicted_with = extract_boxed_answer(response_with) if response_with else ""
        
        results.append({
            'problem': problem,
            'ground_truth': ground_truth,
            'level': level,
            'type': type_,
            'response_without_instruction': response_without,
            'response_with_instruction': response_with,
            'predicted_answer_without': predicted_without,
            'predicted_answer_with': predicted_with,
            'correct_without_instruction': correct_without,
            'correct_with_instruction': correct_with,
            'improvement': int(correct_with) - int(correct_without)
        })
        
        print(f"  Ground truth: {ground_truth}")
        print(f"  Predicted (without): {predicted_without}")
        print(f"  Predicted (with): {predicted_with}")
        print(f"  Correctness: without={correct_without}, with={correct_with}")
        
        # Save intermediate results every 10 questions
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('temp_math_comparison.csv', index=False)
            print(f"  Saved intermediate results: {len(results)} questions processed\n")
    
    return pd.DataFrame(results)

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(results_df: pd.DataFrame) -> Dict:
    """
    Analyze the comparison results.
    
    Args:
        results_df: DataFrame with results
        
    Returns:
        Dictionary with analysis metrics
    """
    print("\nAnalyzing results...")
    
    total = len(results_df)
    correct_without = results_df['correct_without_instruction'].sum()
    correct_with = results_df['correct_with_instruction'].sum()
    
    analysis = {
        'total_questions': total,
        'correct_without_instruction': int(correct_without),
        'correct_with_instruction': int(correct_with),
        'accuracy_without_instruction': correct_without / total if total > 0 else 0,
        'accuracy_with_instruction': correct_with / total if total > 0 else 0,
        'improvement_count': results_df[results_df['improvement'] > 0].shape[0],
        'degradation_count': results_df[results_df['improvement'] < 0].shape[0],
        'no_change_count': results_df[results_df['improvement'] == 0].shape[0],
    }
    
    # Analysis by level
    for level in sorted(results_df['level'].unique()):
        subset = results_df[results_df['level'] == level]
        total_level = len(subset)
        analysis[f'level_{level}_count'] = total_level
        analysis[f'level_{level}_correct_without'] = int(subset['correct_without_instruction'].sum())
        analysis[f'level_{level}_correct_with'] = int(subset['correct_with_instruction'].sum())
        analysis[f'level_{level}_accuracy_without'] = subset['correct_without_instruction'].mean()
        analysis[f'level_{level}_accuracy_with'] = subset['correct_with_instruction'].sum() / total_level if total_level > 0 else 0
    
    # Analysis by type
    for type_ in sorted(results_df['type'].unique()):
        subset = results_df[results_df['type'] == type_]
        total_type = len(subset)
        analysis[f'type_{type_}_count'] = total_type
        analysis[f'type_{type_}_correct_without'] = int(subset['correct_without_instruction'].sum())
        analysis[f'type_{type_}_correct_with'] = int(subset['correct_with_instruction'].sum())
        analysis[f'type_{type_}_accuracy_without'] = subset['correct_without_instruction'].mean()
        analysis[f'type_{type_}_accuracy_with'] = subset['correct_with_instruction'].sum() / total_type if total_type > 0 else 0
    
    return analysis

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results_df: pd.DataFrame, analysis: Dict):
    """
    Create simple distribution chart comparing correctness with and without instruction.
    Similar to create_distribution_charts.py style.
    
    Args:
        results_df: DataFrame with results
        analysis: Dictionary with analysis metrics
    """
    print("\nCreating visualization...")
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create single figure
    plt.figure(figsize=(10, 6))
    
    # Count correct/incorrect for each condition
    without_correct = analysis['correct_without_instruction']
    without_incorrect = analysis['total_questions'] - without_correct
    with_correct = analysis['correct_with_instruction']
    with_incorrect = analysis['total_questions'] - with_correct
    
    # Create side-by-side bars
    x = np.arange(2)
    width = 0.4
    
    plt.bar(x - width/2, [without_correct, without_incorrect], width, 
            label='Without Instruction', alpha=0.8, color='tomato', edgecolor='black')
    plt.bar(x + width/2, [with_correct, with_incorrect], width, 
            label='With Instruction', alpha=0.85, color='steelblue', edgecolor='black')
    
    plt.xlabel('Correctness', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    plt.title(f'Distribution Comparison: Without vs With Instruction\nModel: {MODEL_NAME}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, ['Correct', 'Incorrect'])
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip([without_correct, without_incorrect], 
                                      [with_correct, with_incorrect])):
        if v1 > 0:
            plt.text(i - width/2, v1 + 0.05, str(v1), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        if v2 > 0:
            plt.text(i + width/2, v2 + 0.05, str(v2), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'math_dataset_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{output_path}'")
    
    plt.show()

# ============================================================================
# REPORTING
# ============================================================================

def print_summary(analysis: Dict):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total questions analyzed: {analysis['total_questions']}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Instruction: {INSTRUCTION}")
    print()
    
    print("Overall Performance:")
    print(f"  Without Instruction:")
    print(f"    - Correct: {analysis['correct_without_instruction']}")
    print(f"    - Accuracy: {analysis['accuracy_without_instruction']*100:.2f}%")
    print()
    print(f"  With Instruction:")
    print(f"    - Correct: {analysis['correct_with_instruction']}")
    print(f"    - Accuracy: {analysis['accuracy_with_instruction']*100:.2f}%")
    print()
    print(f"Performance Changes:")
    print(f"  - Improved: {analysis['improvement_count']} questions")
    print(f"  - No Change: {analysis['no_change_count']} questions")
    print(f"  - Degraded: {analysis['degradation_count']} questions")
    print()
    
    # Print by level
    print("Analysis by Difficulty Level:")
    levels = sorted([k.replace('level_', '').replace('_count', '') 
                     for k in analysis.keys() if k.startswith('level_') and k.endswith('_count')])
    for level in levels:
        count = analysis.get(f'level_{level}_count', 0)
        correct_without = analysis.get(f'level_{level}_correct_without', 0)
        correct_with = analysis.get(f'level_{level}_correct_with', 0)
        acc_without = analysis.get(f'level_{level}_accuracy_without', 0)
        acc_with = analysis.get(f'level_{level}_accuracy_with', 0)
        print(f"  {level} ({count} questions):")
        print(f"    - Without: {correct_without}/{count} ({acc_without*100:.1f}%)")
        print(f"    - With: {correct_with}/{count} ({acc_with*100:.1f}%)")
    print()
    
    # Print by type
    print("Analysis by Problem Type:")
    types = sorted([k.replace('type_', '').replace('_count', '') 
                    for k in analysis.keys() if k.startswith('type_') and k.endswith('_count')])
    for type_ in types:
        count = analysis.get(f'type_{type_}_count', 0)
        correct_without = analysis.get(f'type_{type_}_correct_without', 0)
        correct_with = analysis.get(f'type_{type_}_correct_with', 0)
        acc_without = analysis.get(f'type_{type_}_accuracy_without', 0)
        acc_with = analysis.get(f'type_{type_}_accuracy_with', 0)
        print(f"  {type_} ({count} questions):")
        print(f"    - Without: {correct_without}/{count} ({acc_without*100:.1f}%)")
        print(f"    - With: {correct_with}/{count} ({acc_with*100:.1f}%)")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("MATH Dataset Instruction Comparison")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Number of questions: {NUM_QUESTIONS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load dataset
    questions_df = load_math_dataset(num_samples=NUM_QUESTIONS)
    
    # Run comparison
    results_df = run_comparison(questions_df)
    
    # Save results as CSV
    output_csv = 'math_dataset_comparison_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")
    
    # Save detailed results as JSON
    output_json = 'math_dataset_comparison_results.json'
    results_df.to_json(output_json, orient='records', indent=2)
    print(f"Detailed results saved to '{output_json}'")
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    # Save analysis
    analysis_json = 'math_dataset_comparison_analysis.json'
    with open(analysis_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to '{analysis_json}'")
    
    # Print summary
    print_summary(analysis)
    
    # Create visualizations
    create_visualizations(results_df, analysis)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {output_csv} (results in CSV format)")
    print(f"  - {output_json} (results in JSON format)")
    print(f"  - {analysis_json} (analysis metrics)")
    print(f"  - math_dataset_comparison.png (visualization)")
    print("="*70)

if __name__ == "__main__":
    main()
