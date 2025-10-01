#!/usr/bin/env python3
"""
Script to compare math problem performance with and without instructions.
Samples 100 random questions from the MATH dataset on HuggingFace,
tests them with OpenRouter API, and visualizes the comparison.
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
MODEL_NAME = os.getenv('MODEL_NAME', 'openai/gpt-4o-mini')

# Instruction to use for the "with instruction" condition
INSTRUCTION = """roduce final output? Actually the conversation: user gave a huge prompt with several seemingly 
random text that appears to be a mixture of instructions, but final part is many sample 
inputs/outputs. The instruction at the beginning: "Write a general-purpose program that can solve 
any of these problems." Then "The assistant response should be a single line with your answer to the 
following prompt:" Then the prompt is an input with vector v etc. We need produce a single line with 
the answer. So it\'s the final prompt: "Input: There are an infinite number ... find the vector v 
that has smallest magnitude. Output: ..." That is provided. So"""

# Number of questions to sample
NUM_QUESTIONS = int(os.getenv('NUM_QUESTIONS', '100'))

# API retry configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_BACKOFF_SEC = float(os.getenv('RETRY_BACKOFF_SEC', '2.0'))
REQUEST_DELAY_SEC = float(os.getenv('REQUEST_DELAY_SEC', '0.5'))

# Random seed for reproducibility
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

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
# DATA LOADING
# ============================================================================

def load_math_dataset(num_samples: int = 100) -> pd.DataFrame:
    """
    Load MATH dataset from HuggingFace and randomly sample Level 5 questions.
    
    Args:
        num_samples: Number of questions to randomly sample
        
    Returns:
        DataFrame with columns: problem, solution, level, type
    """
    print(f"Loading MATH dataset from HuggingFace...")
    
    try:
        # Load the dataset
        dataset = load_dataset("qwedsacf/competition_math", split="train")
        print(f"Total questions in dataset: {len(dataset)}")
        
        # Filter for Level 5 questions only
        level_5_questions = [item for item in dataset if item.get('level') == 'Level 5']
        print(f"Level 5 questions available: {len(level_5_questions)}")
        
        # Randomly sample from Level 5 questions
        if num_samples > len(level_5_questions):
            print(f"Warning: Requested {num_samples} samples but only {len(level_5_questions)} Level 5 questions available")
            num_samples = len(level_5_questions)
        
        sampled = random.sample(level_5_questions, num_samples)
        print(f"Randomly sampled {len(sampled)} Level 5 questions")
        
        # Convert to DataFrame
        data = []
        for item in sampled:
            data.append({
                'problem': item['problem'],
                'solution': item['solution'],
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

# ============================================================================
# EVALUATION
# ============================================================================

def run_comparison(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run comparison: test each question with and without instruction.
    
    Args:
        questions_df: DataFrame with questions
        
    Returns:
        DataFrame with results
    """
    print(f"\nRunning comparison on {len(questions_df)} questions...")
    print(f"Model: {MODEL_NAME}")
    print(f"Instruction: {INSTRUCTION[:100]}...")
    print()
    
    results = []
    
    for idx, row in questions_df.iterrows():
        problem = row['problem']
        solution = row['solution']
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
        
        # Calculate response lengths
        len_without = len(response_without)
        len_with = len(response_with)
        
        results.append({
            'problem': problem,
            'solution': solution,
            'level': level,
            'type': type_,
            'response_without_instruction': response_without,
            'response_with_instruction': response_with,
            'length_without': len_without,
            'length_with': len_with,
            'length_diff': len_with - len_without
        })
        
        print(f"  Response lengths: without={len_without}, with={len_with}, diff={len_with - len_without}")
        
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
    
    analysis = {
        'total_questions': len(results_df),
        'avg_length_without': results_df['length_without'].mean(),
        'avg_length_with': results_df['length_with'].mean(),
        'avg_length_diff': results_df['length_diff'].mean(),
        'median_length_without': results_df['length_without'].median(),
        'median_length_with': results_df['length_with'].median(),
        'std_length_without': results_df['length_without'].std(),
        'std_length_with': results_df['length_with'].std(),
    }
    
    # Analysis by level
    for level in results_df['level'].unique():
        subset = results_df[results_df['level'] == level]
        analysis[f'level_{level}_count'] = len(subset)
        analysis[f'level_{level}_avg_length_without'] = subset['length_without'].mean()
        analysis[f'level_{level}_avg_length_with'] = subset['length_with'].mean()
        analysis[f'level_{level}_avg_diff'] = subset['length_diff'].mean()
    
    # Analysis by type
    for type_ in results_df['type'].unique():
        subset = results_df[results_df['type'] == type_]
        analysis[f'type_{type_}_count'] = len(subset)
        analysis[f'type_{type_}_avg_length_without'] = subset['length_without'].mean()
        analysis[f'type_{type_}_avg_length_with'] = subset['length_with'].mean()
        analysis[f'type_{type_}_avg_diff'] = subset['length_diff'].mean()
    
    return analysis

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results_df: pd.DataFrame, analysis: Dict):
    """
    Create comprehensive visualizations comparing performance.
    
    Args:
        results_df: DataFrame with results
        analysis: Dictionary with analysis metrics
    """
    print("\nCreating visualizations...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'MATH Dataset Comparison: With vs Without Instruction\nModel: {MODEL_NAME}', 
                 fontsize=16, fontweight='bold')
    
    # ========================================================================
    # 1. Response Length Comparison (Box Plot)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    data_for_box = [
        results_df['length_without'].values,
        results_df['length_with'].values
    ]
    bp = ax1.boxplot(data_for_box, labels=['Without Instruction', 'With Instruction'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax1.set_ylabel('Response Length (characters)')
    ax1.set_title('Response Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. Average Response Length Comparison (Bar Chart)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Without\nInstruction', 'With\nInstruction']
    averages = [analysis['avg_length_without'], analysis['avg_length_with']]
    colors = ['lightcoral', 'lightblue']
    bars = ax2.bar(categories, averages, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Response Length (characters)')
    ax2.set_title('Average Response Length')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, avg in zip(bars, averages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{avg:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # 3. Length Difference Distribution (Histogram)
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(results_df['length_diff'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax3.axvline(x=analysis['avg_length_diff'], color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {analysis["avg_length_diff"]:.0f}')
    ax3.set_xlabel('Length Difference (with - without)')
    ax3.set_ylabel('Number of Questions')
    ax3.set_title('Distribution of Length Differences')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # 4. Response Length by Difficulty Level
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    levels = sorted(results_df['level'].unique())
    x = np.arange(len(levels))
    width = 0.35
    
    avg_without_by_level = [results_df[results_df['level'] == level]['length_without'].mean() 
                            for level in levels]
    avg_with_by_level = [results_df[results_df['level'] == level]['length_with'].mean() 
                         for level in levels]
    
    ax4.bar(x - width/2, avg_without_by_level, width, label='Without Instruction', 
            alpha=0.7, color='lightcoral')
    ax4.bar(x + width/2, avg_with_by_level, width, label='With Instruction', 
            alpha=0.7, color='lightblue')
    ax4.set_xlabel('Difficulty Level')
    ax4.set_ylabel('Average Response Length')
    ax4.set_title('Response Length by Difficulty Level')
    ax4.set_xticks(x)
    ax4.set_xticklabels(levels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 5. Response Length by Problem Type
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    types = sorted(results_df['type'].unique())
    x = np.arange(len(types))
    
    avg_without_by_type = [results_df[results_df['type'] == t]['length_without'].mean() 
                           for t in types]
    avg_with_by_type = [results_df[results_df['type'] == t]['length_with'].mean() 
                        for t in types]
    
    ax5.bar(x - width/2, avg_without_by_type, width, label='Without Instruction', 
            alpha=0.7, color='lightcoral')
    ax5.bar(x + width/2, avg_with_by_type, width, label='With Instruction', 
            alpha=0.7, color='lightblue')
    ax5.set_xlabel('Problem Type')
    ax5.set_ylabel('Average Response Length')
    ax5.set_title('Response Length by Problem Type')
    ax5.set_xticks(x)
    ax5.set_xticklabels(types, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 6. Scatter Plot: Length Without vs With Instruction
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(results_df['length_without'], results_df['length_with'], 
                alpha=0.5, c='purple', s=30)
    
    # Add diagonal line (y=x)
    max_val = max(results_df['length_without'].max(), results_df['length_with'].max())
    ax6.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal length')
    
    ax6.set_xlabel('Length Without Instruction')
    ax6.set_ylabel('Length With Instruction')
    ax6.set_title('Response Length Correlation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ========================================================================
    # 7. Sample Distribution by Level
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    level_counts = results_df['level'].value_counts().sort_index()
    ax7.bar(range(len(level_counts)), level_counts.values, color='skyblue', alpha=0.7)
    ax7.set_xlabel('Difficulty Level')
    ax7.set_ylabel('Number of Questions')
    ax7.set_title('Sample Distribution by Level')
    ax7.set_xticks(range(len(level_counts)))
    ax7.set_xticklabels(level_counts.index)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, count in enumerate(level_counts.values):
        ax7.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # 8. Sample Distribution by Type
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    type_counts = results_df['type'].value_counts()
    ax8.barh(range(len(type_counts)), type_counts.values, color='lightgreen', alpha=0.7)
    ax8.set_yticks(range(len(type_counts)))
    ax8.set_yticklabels(type_counts.index)
    ax8.set_xlabel('Number of Questions')
    ax8.set_title('Sample Distribution by Type')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, count in enumerate(type_counts.values):
        ax8.text(count + 0.5, i, str(count), ha='left', va='center', fontweight='bold')
    
    # ========================================================================
    # 9. Summary Statistics Table
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Total Questions: {analysis['total_questions']}
    
    Response Length (characters):
    • Without Instruction:
      - Mean: {analysis['avg_length_without']:.0f}
      - Median: {analysis['median_length_without']:.0f}
      - Std Dev: {analysis['std_length_without']:.0f}
    
    • With Instruction:
      - Mean: {analysis['avg_length_with']:.0f}
      - Median: {analysis['median_length_with']:.0f}
      - Std Dev: {analysis['std_length_with']:.0f}
    
    • Difference (With - Without):
      - Mean: {analysis['avg_length_diff']:.0f}
    
    Instruction Used:
    {INSTRUCTION[:150]}...
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
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
    print(f"Instruction: {INSTRUCTION}")
    print()
    
    print("Response Length Statistics:")
    print(f"  Without Instruction:")
    print(f"    - Mean: {analysis['avg_length_without']:.2f} characters")
    print(f"    - Median: {analysis['median_length_without']:.2f} characters")
    print(f"    - Std Dev: {analysis['std_length_without']:.2f} characters")
    print()
    print(f"  With Instruction:")
    print(f"    - Mean: {analysis['avg_length_with']:.2f} characters")
    print(f"    - Median: {analysis['median_length_with']:.2f} characters")
    print(f"    - Std Dev: {analysis['std_length_with']:.2f} characters")
    print()
    print(f"  Average Difference (With - Without): {analysis['avg_length_diff']:.2f} characters")
    print()
    
    # Print by level
    print("Analysis by Difficulty Level:")
    levels = sorted([k.replace('level_', '').replace('_count', '') 
                     for k in analysis.keys() if k.startswith('level_') and k.endswith('_count')])
    for level in levels:
        count = analysis.get(f'level_{level}_count', 0)
        avg_without = analysis.get(f'level_{level}_avg_length_without', 0)
        avg_with = analysis.get(f'level_{level}_avg_length_with', 0)
        avg_diff = analysis.get(f'level_{level}_avg_diff', 0)
        print(f"  Level {level} ({count} questions):")
        print(f"    - Avg length without: {avg_without:.2f}")
        print(f"    - Avg length with: {avg_with:.2f}")
        print(f"    - Avg difference: {avg_diff:.2f}")
    print()
    
    # Print by type
    print("Analysis by Problem Type:")
    types = sorted([k.replace('type_', '').replace('_count', '') 
                    for k in analysis.keys() if k.startswith('type_') and k.endswith('_count')])
    for type_ in types:
        count = analysis.get(f'type_{type_}_count', 0)
        avg_without = analysis.get(f'type_{type_}_avg_length_without', 0)
        avg_with = analysis.get(f'type_{type_}_avg_length_with', 0)
        avg_diff = analysis.get(f'type_{type_}_avg_diff', 0)
        print(f"  {type_} ({count} questions):")
        print(f"    - Avg length without: {avg_without:.2f}")
        print(f"    - Avg length with: {avg_with:.2f}")
        print(f"    - Avg difference: {avg_diff:.2f}")
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
    print(f"Number of questions: {NUM_QUESTIONS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load dataset
    questions_df = load_math_dataset(num_samples=NUM_QUESTIONS)
    
    # Run comparison
    results_df = run_comparison(questions_df)
    
    # Save results
    output_csv = 'math_dataset_comparison_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")
    
    # Save detailed results with full responses
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

if __name__ == "__main__":
    main()
