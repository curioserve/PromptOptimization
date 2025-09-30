# MATH500 Dataset Evaluation

This folder contains evaluation tools for the MATH500 dataset. It supports:
- Transformers pipeline (local/hosted Hugging Face models)
- OpenRouter Chat Completions API (set OPENROUTER_API_KEY)

## Dataset Format
The MATH500 dataset contains mathematical problems with the following structure:
- `problem`: The mathematical problem statement
- `solution`: Detailed solution with reasoning
- `answer`: Final answer (extracted value)
- `subject`: Mathematical subject area
- `level`: Difficulty level (1-5)
- `unique_id`: Unique identifier

## Usage

### Transformers (Hugging Face) example
```bash
python MATH500/run_math500_evaluation.py \
  --model ./gpt-oss-20b \
  --samples 50 \
  --runs 3 \
  --max_tokens 1024 \
  --output math500_results.json \
  --provider transformers \
  --verbose
```

### OpenRouter example
```bash
export OPENROUTER_API_KEY="YOUR_KEY"
python MATH500/run_math500_evaluation.py \
  --model openai/gpt-4o-mini \
  --samples 50 \
  --runs 1 \
  --max_tokens 1024 \
  --output math500_results.json \
  --provider openrouter \
  --verbose
```

### Command Line Options
- `--model`: Model path or id. For OpenRouter, pass the OpenRouter model id (e.g., `openai/gpt-4o-mini`). Default: `./gpt-oss-20b`.
- `--samples`: Number of samples to evaluate (use `None` for all)
- `--runs`: Number of evaluation runs for statistics
- `--max_tokens`: Maximum new tokens to generate
- `--output`: Output JSON file path (main consolidated results)
- `--verbose`: Enable detailed logging
- `--subject`: Filter by subject (e.g., Algebra, Geometry)
- `--provider`: `transformers` or `openrouter` (default: `transformers`)
- `--openrouter_api_key`: OpenRouter API key (optional; otherwise use env var `OPENROUTER_API_KEY`)

## Output
Running the script writes two kinds of outputs:

- Main JSON (path from `--output`):
  - Experiment metadata (model, tokens, subject filter, counts)
  - Overall statistics (accuracy and timing across runs)
  - `individual_runs`: each run includes its `summary` and per-sample `results`

- Experiment folder: `results/math500_experiment_<timestamp>/`
  - `complete_results.json`: same core data as the main JSON, saved under the experiment folder
  - `individual_runs/run_XX_detailed.json`: per-run detailed JSON
  - `individual_runs/run_XX_results.csv`: per-run CSV
  - `cross_run_comparison.csv`: per-question correctness and answers across runs
  - `inconsistent_questions.csv`: only if multiple runs and inconsistencies exist
  - `summary_statistics.json`: aggregated analysis across runs
  - `subject_analysis.csv`, `level_analysis.csv`: subject/level performance breakdowns

### Per-sample fields
Each sample entry includes:
- `problem`, `ground_truth`, `generated_text`
- `reasoning_chain` (duplicate of `generated_text`)
- `predicted_answer`, `is_correct`, `inference_time`, `subject`, `level`

For `--provider openrouter`, a `reasoning_meta` object may be included with safe metadata. Raw private chain-of-thought is not stored. If enabled in code (`save_reasoning_summary=True`), a high-level `reasoning_summary` may be included.

Note: Additional OpenRouter reasoning controls (`reasoning_effort`, `reasoning_max_tokens`, `reasoning_exclude`, `reasoning_enabled`) exist in code but are not exposed via CLI.
