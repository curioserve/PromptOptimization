# DAPO Math Dataset Evaluation

This project evaluates the **OpenAI GPT-OSS-20B** model on the **DAPO Math 17k Processed** dataset from HuggingFace.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run a quick test (3 samples):**
```bash
python quick_test.py
```

3. **Run full evaluation:**
```bash
python run_dapo_evaluation.py --samples 50
```

4. **Run evaluation with specific dataset config:**
```bash
python run_dapo_evaluation.py --samples 50 --dataset_config en
```

## Usage Examples

### Basic Evaluation
```bash
# Evaluate 10 samples (default)
python run_dapo_evaluation.py

# Evaluate 100 samples
python run_dapo_evaluation.py --samples 100

# Evaluate all samples (17k+)
python run_dapo_evaluation.py --samples -1
```

### Advanced Options
```bash
# Use different model
python run_dapo_evaluation.py --model "microsoft/DialoGPT-large" --samples 20

# Increase token limit
python run_dapo_evaluation.py --max_tokens 1024 --samples 50

# Custom output file
python run_dapo_evaluation.py --output my_results.json --verbose

# Use English-only dataset
python run_dapo_evaluation.py --dataset_config en --samples 100

# Use Chinese-only dataset
python run_dapo_evaluation.py --dataset_config cn --samples 100
```

### Programmatic Usage
```python
from dapo_evaluator import DAPOEvaluator

# Initialize evaluator (default uses 'all' config)
evaluator = DAPOEvaluator(model_id="openai/gpt-oss-20b")

# Initialize evaluator with specific dataset config
evaluator = DAPOEvaluator(
    model_id="openai/gpt-oss-20b",
    dataset_config="en"  # or "cn" or "all"
)

# Run evaluation
summary = evaluator.evaluate_dataset(num_samples=20)

# Save results
evaluator.save_results("results.json")

# Print sample results
evaluator.print_sample_results()
```

## Dataset Format

The DAPO dataset contains math problems with:
- **prompt**: The math problem statement
- **solution**: Ground truth answer (usually numeric)
- **source_prompt**: Formatted messages for the model
- **data_source**: Source identifier (e.g., "math_dapo")
- **ability**: Problem category (e.g., "MATH")

Example:
```json
{
  "prompt": "In triangle ABC, sin∠A = 4/5...",
  "solution": "34",
  "source_prompt": [{"role": "user", "content": "Solve the following..."}],
  "data_source": "math_dapo",
  "ability": "MATH"
}
```

## Output Format

Results are saved as JSON with:
```json
{
  "summary": {
    "total_samples": 50,
    "correct_predictions": 23,
    "accuracy": 0.46,
    "model_id": "openai/gpt-oss-20b"
  },
  "detailed_results": [
    {
      "prompt": "...",
      "ground_truth": "34",
      "predicted_answer": "34",
      "is_correct": true,
      "inference_time": 2.3
    }
  ]
}
```

## Evaluation Metrics

- **Accuracy**: Percentage of correct predictions
- **Inference Time**: Average time per sample
- **Answer Extraction**: Robust parsing of model outputs

## Requirements

- Python 3.8+
- PyTorch 2.2+
- Transformers 4.35+
- HuggingFace Datasets
- CUDA-compatible GPU (recommended)

## Files

- `dapo_evaluator.py`: Main evaluation class
- `run_dapo_evaluation.py`: Command-line interface
- `quick_test.py`: Quick test with 3 samples
- `requirements.txt`: Python dependencies
