import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
CSV_PATH = '/Users/ali/Documents/Hob/projects/PromptOptimization/math_instruction_results.csv'
OUT_DIR = '/Users/ali/Documents/Hob/projects/PromptOptimization'

# Read the CSV file
df = pd.read_csv(CSV_PATH)

# Ensure expected columns exist
required_cols = {'original_correct_count', 'correct_count_with_instruction'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

# Clean and coerce to integers within 0..5 range
df['original_correct_count'] = pd.to_numeric(df['original_correct_count'], errors='coerce').fillna(0).astype(int).clip(0, 5)
df['correct_count_with_instruction'] = pd.to_numeric(df['correct_count_with_instruction'], errors='coerce').fillna(0).astype(int).clip(0, 5)

# Build counts for 0..5
bins = list(range(0, 6))
orig_counts = df['original_correct_count'].value_counts().reindex(bins, fill_value=0)
instr_counts = df['correct_count_with_instruction'].value_counts().reindex(bins, fill_value=0)

# Chart 1: Distribution without instruction
plt.figure(figsize=(8, 6))
plt.bar(orig_counts.index, orig_counts.values, color='tomato', alpha=0.8, edgecolor='black')
plt.xlabel('Number of Correct Answers (out of 5)')
plt.ylabel('Number of Questions')
plt.title('Distribution of Questions by Correct Answers (Without Instruction)')
plt.xticks(bins, [f'{b}/5' for b in bins])
plt.grid(True, axis='y', alpha=0.3)
for x, v in zip(orig_counts.index, orig_counts.values):
    if v > 0:
        plt.text(x, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/without_instruction_distribution.png', dpi=300, bbox_inches='tight')

# Chart 2: Distribution with instruction
plt.figure(figsize=(8, 6))
plt.bar(instr_counts.index, instr_counts.values, color='steelblue', alpha=0.85, edgecolor='black')
plt.xlabel('Number of Correct Answers (out of 5)')
plt.ylabel('Number of Questions')
plt.title('Distribution of Questions by Correct Answers (With Instruction)')
plt.xticks(bins, [f'{b}/5' for b in bins])
plt.grid(True, axis='y', alpha=0.3)
for x, v in zip(instr_counts.index, instr_counts.values):
    if v > 0:
        plt.text(x, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/with_instruction_distribution.png', dpi=300, bbox_inches='tight')

# Combined side-by-side (optional convenience)
plt.figure(figsize=(10, 6))
x = np.arange(len(bins))
width = 0.4
plt.bar(x - width/2, orig_counts.values, width, label='Without Instruction', color='tomato', alpha=0.8)
plt.bar(x + width/2, instr_counts.values, width, label='With Instruction', color='steelblue', alpha=0.85)
plt.xlabel('Number of Correct Answers (out of 5)')
plt.ylabel('Number of Questions')
plt.title('Distribution Comparison: Without vs With Instruction')
plt.xticks(x, [f'{b}/5' for b in bins])
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
for i, (v1, v2) in enumerate(zip(orig_counts.values, instr_counts.values)):
    if v1 > 0:
        plt.text(i - width/2, v1 + 0.05, str(v1), ha='center', va='bottom', fontsize=8)
    if v2 > 0:
        plt.text(i + width/2, v2 + 0.05, str(v2), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/question_distribution_charts_v2.png', dpi=300, bbox_inches='tight')

print('Saved:')
print(f' - {OUT_DIR}/without_instruction_distribution.png')
print(f' - {OUT_DIR}/with_instruction_distribution.png')
print(f' - {OUT_DIR}/question_distribution_charts_v2.png')
