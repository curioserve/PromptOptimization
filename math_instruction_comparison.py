#!/usr/bin/env python3
"""
Script to compare math problem performance with and without instructions.
Filters questions answered correctly 2, 3, or 4 times out of 5 runs,
then tests them with instruction using OpenRouter GPT-OSS-20B.
"""

import pandas as pd
import os
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from openai import OpenAI

# Load environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', "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")
EVAL_API_MODEL = os.getenv('EVAL_API_MODEL', "openai/gpt-oss-20b")
EVAL_BATCH_SIZE = int(os.getenv('EVAL_BATCH_SIZE', '10'))

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Instruction to be used
INSTRUCTION = """produce final output? Actually the conversation: user gave a huge prompt with several seemingly random text that appears to be a mixture of instructions, but final part is many sample inputs/outputs. Th
e instruction at the beginning: "Write a general-purpose program that can solve any of these problems." Then "The assistant response should be a single line with your answer to the following prompt:" Then
the prompt is an input with vector v etc. We need produce a single line with the answer. So it\'s the final prompt: "Input: There are an infinite number ... find the vector v that has smallest magnitude. O
utput: ..." That is provided. So"""

def load_and_filter_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and filter questions answered correctly 2, 3, or 4 times."""
    print("Loading and filtering data...")
    df = pd.read_csv(csv_path)
    
    # Filter questions with correct_count in [2, 3, 4]
    filtered_df = df[df['correct_count'].isin([2, 3, 4])].copy()
    
    print(f"Total questions: {len(df)}")
    print(f"Questions answered correctly 2-4 times: {len(filtered_df)}")
    print(f"Distribution of correct counts:")
    print(filtered_df['correct_count'].value_counts().sort_index())
    
    return filtered_df

def test_api_connection() -> bool:
    """Test the OpenRouter API connection."""
    print("Testing API connection...")
    
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
        print(f"API test failed with exception: {e}")
        return False

def query_model_with_instruction(problem: str, ground_truth: str, run_number: int = 1) -> Tuple[str, bool]:
    """Query the model with instruction and return response and correctness."""
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
        
        # Simple correctness check - you might want to make this more sophisticated
        is_correct = ground_truth.lower().strip() in answer.lower() or answer.lower().strip() in ground_truth.lower()
        
        return answer, is_correct
        
    except Exception as e:
        print(f"API call failed with exception: {e} (run {run_number})")
        return "", False

def run_evaluation(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Run evaluation on filtered questions with instruction - 5 times each."""
    print(f"Running evaluation on {len(filtered_df)} questions (5 iterations each)...")
    
    results = []
    
    for idx, row in filtered_df.iterrows():
        problem = row['problem']
        ground_truth = str(row['ground_truth'])
        original_correct_count = row['correct_count']
        
        print(f"Processing question {idx + 1}/{len(filtered_df)}")
        
        # Run each question 5 times with instruction
        correct_count_with_instruction = 0
        responses_with_instruction = []
        
        for run in range(1, 6):
            print(f"  Run {run}/5")
            response, is_correct = query_model_with_instruction(problem, ground_truth, run)
            responses_with_instruction.append(response)
            if is_correct:
                correct_count_with_instruction += 1
            
            # Rate limiting between runs
            time.sleep(1)
        
        results.append({
            'problem': problem,
            'ground_truth': ground_truth,
            'original_correct_count': original_correct_count,
            'correct_count_with_instruction': correct_count_with_instruction,
            'responses_with_instruction': responses_with_instruction,
            'improvement': correct_count_with_instruction - original_correct_count
        })
        
        print(f"  Original: {original_correct_count}/5, With instruction: {correct_count_with_instruction}/5")
        
        # Save intermediate results every 5 questions
        if len(results) % 5 == 0:
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
    for count in [2, 3, 4]:
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
    
    # Ensure all counts 0-5 are represented
    all_counts = range(0, 6)
    orig_values = [original_dist.get(i, 0) for i in all_counts]
    inst_values = [instruction_dist.get(i, 0) for i in all_counts]
    
    x = np.arange(len(all_counts))
    width = 0.35
    
    ax1.bar(x - width/2, orig_values, width, label='Without Instruction', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, inst_values, width, label='With Instruction', alpha=0.7, color='lightblue')
    ax1.set_xlabel('Number of Correct Answers (out of 5)')
    ax1.set_ylabel('Number of Questions')
    ax1.set_title('Distribution Comparison: Questions by Correct Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i}/5' for i in all_counts])
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
    ax2.set_ylabel('Average Correct Answers (out of 5)')
    ax2.set_title('Average Performance Comparison')
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, avg in zip(bars, averages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{avg:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Improvement by original performance level
    improvement_by_level = []
    labels = []
    for count in [2, 3, 4]:
        subset = results_df[results_df['original_correct_count'] == count]
        if len(subset) > 0:
            avg_improvement = subset['improvement'].mean()
            improvement_by_level.append(avg_improvement)
            labels.append(f'Originally {count}/5')
        else:
            improvement_by_level.append(0)
            labels.append(f'Originally {count}/5')
    
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
    print(f"Average correct without instruction: {analysis['avg_original_correct']:.2f}/5")
    print(f"Average correct with instruction: {analysis['avg_instruction_correct']:.2f}/5")
    print(f"Average improvement: {analysis['avg_improvement']:.2f}")
    print()
    
    print("Distribution without instruction:")
    for count, freq in analysis['original_distribution'].items():
        print(f"  {count}/5 correct: {freq} questions")
    
    print("\nDistribution with instruction:")
    for count, freq in analysis['instruction_distribution'].items():
        print(f"  {count}/5 correct: {freq} questions")
    
    print("\nImprovement by original performance:")
    for count in [2, 3, 4]:
        key = f'questions_originally_{count}_correct'
        if key in analysis:
            total = analysis[key]
            avg_perf = analysis[f'avg_performance_with_instruction_from_{count}']
            avg_imp = analysis[f'avg_improvement_from_{count}']
            print(f"Questions originally {count}/5: {total} questions")
            print(f"  - Average performance with instruction: {avg_perf:.2f}/5")
            print(f"  - Average improvement: {avg_imp:.2f}")
            print()

def main():
    """Main execution function."""
    print("Starting Math Instruction Comparison Analysis")
    print("=" * 50)
    
    # Load and filter data
    csv_path = "/Users/ali/Documents/Hob/projects/PromptOptimization/Results/math500_20b_results_summary.csv"
    filtered_df = load_and_filter_data(csv_path)
    
    # Test API connection
    if not test_api_connection():
        print("API connection failed. Please check your credentials.")
        return
    
    # For testing, let's start with a smaller sample
    print(f"\nStarting with first 20 questions for testing...")
    sample_df = filtered_df.head(20).copy()
    
    # Run evaluation
    results_df = run_evaluation(sample_df)
    
    # Save results
    results_df.to_csv('math_instruction_results.csv', index=False)
    print("Results saved to 'math_instruction_results.csv'")
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    # Create visualization
    create_visualization(results_df, analysis)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
