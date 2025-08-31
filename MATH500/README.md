# MATH500 Dataset Evaluation

This folder contains evaluation tools for the MATH500 dataset using the GPT-OSS-20B model.

## Dataset Format
The MATH500 dataset contains mathematical problems with the following structure:
- `problem`: The mathematical problem statement
- `solution`: Detailed solution with reasoning
- `answer`: Final answer (extracted value)
- `subject`: Mathematical subject area
- `level`: Difficulty level (1-5)
- `unique_id`: Unique identifier

## Usage

### Quick Test
```bash
python quick_test.py
```

### Full Evaluation
```bash
python run_math500_evaluation.py --samples 50 --runs 3 --verbose
```

### Command Line Options
- `--model`: Model path (default: "./gpt-oss-20b")
- `--samples`: Number of samples to evaluate
- `--runs`: Number of evaluation runs for statistics
- `--max_tokens`: Maximum tokens to generate
- `--output`: Output JSON file path
- `--verbose`: Enable detailed logging
- `--subject`: Filter by subject (optional)

## Output
Results are saved in JSON format with:
- Overall statistics across multiple runs
- Individual run details
- Per-sample results with reasoning chains
