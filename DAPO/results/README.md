# DAPO Evaluation Results

This directory contains detailed results from DAPO dataset evaluations.

## Directory Structure

Each experiment creates a timestamped folder: `dapo_experiment_YYYYMMDD_HHMMSS/`

### Files Generated Per Experiment:

#### Main Results
- `complete_results.json` - Comprehensive experiment data with all runs
- `summary_statistics.json` - Key metrics and statistical analysis

#### Individual Run Data
- `individual_runs/` - Detailed results for each run
  - `run_01_detailed.json` - Complete JSON data for run 1
  - `run_01_results.csv` - CSV format for easy analysis
  - `run_02_detailed.json` - Complete JSON data for run 2
  - `run_02_results.csv` - CSV format for easy analysis
  - ... (one pair per run)

#### Cross-Run Analysis
- `cross_run_comparison.csv` - Table showing correctness per question per run
- `inconsistent_questions.csv` - Questions with different results across runs

## Key Features

### Cross-Run Comparison Table
Shows which questions were answered correctly in each run:
- ✓ = Correct answer
- ✗ = Incorrect answer
- Consistency metrics for each question
- Identifies questions with inconsistent results across runs

### Inconsistent Questions Analysis
Questions where GPT gives different answers across runs are saved separately for analysis.

### Statistical Tracking
- Per-run accuracy and timing
- Mean ± standard deviation across runs
- Min/max performance ranges
- Consistency rate analysis
